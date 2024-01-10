#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <string.h>
#include <assert.h>
#include <errno.h>
#include <math.h>

#include "gguflib.h"
#include "sds.h"
#include "fp16.h"

/* Global options that can could be used for all the subcommands. */
struct {
    int verbose;        // --verbose option
} Opt = {0};

/* ========================== Utility functions  ============================ */

/* Glob-style pattern matching. Return 1 on match, 0 otherwise. */
int strmatch(const char *pattern, int patternLen,
             const char *string, int stringLen, int nocase)
{
    while(patternLen && stringLen) {
        switch(pattern[0]) {
        case '*':
            while (patternLen && pattern[1] == '*') {
                pattern++;
                patternLen--;
            }
            if (patternLen == 1)
                return 1; /* match */
            while(stringLen) {
                if (strmatch(pattern+1, patternLen-1,
                             string, stringLen, nocase))
                    return 1; /* match */
                string++;
                stringLen--;
            }
            return 0; /* no match */
            break;
        case '?':
            string++;
            stringLen--;
            break;
        case '[':
        {
            int not, match;

            pattern++;
            patternLen--;
            not = pattern[0] == '^';
            if (not) {
                pattern++;
                patternLen--;
            }
            match = 0;
            while(1) {
                if (pattern[0] == '\\' && patternLen >= 2) {
                    pattern++;
                    patternLen--;
                    if (pattern[0] == string[0])
                        match = 1;
                } else if (pattern[0] == ']') {
                    break;
                } else if (patternLen == 0) {
                    pattern--;
                    patternLen++;
                    break;
                } else if (patternLen >= 3 && pattern[1] == '-') {
                    int start = pattern[0];
                    int end = pattern[2];
                    int c = string[0];
                    if (start > end) {
                        int t = start;
                        start = end;
                        end = t;
                    }
                    if (nocase) {
                        start = tolower(start);
                        end = tolower(end);
                        c = tolower(c);
                    }
                    pattern += 2;
                    patternLen -= 2;
                    if (c >= start && c <= end)
                        match = 1;
                } else {
                    if (!nocase) {
                        if (pattern[0] == string[0])
                            match = 1;
                    } else {
                        if (tolower((int)pattern[0]) == tolower((int)string[0]))
                            match = 1;
                    }
                }
                pattern++;
                patternLen--;
            }
            if (not)
                match = !match;
            if (!match)
                return 0; /* no match */
            string++;
            stringLen--;
            break;
        }
        case '\\':
            if (patternLen >= 2) {
                pattern++;
                patternLen--;
            }
            /* fall through */
        default:
            if (!nocase) {
                if (pattern[0] != string[0])
                    return 0; /* no match */
            } else {
                if (tolower((int)pattern[0]) != tolower((int)string[0]))
                    return 0; /* no match */
            }
            string++;
            stringLen--;
            break;
        }
        pattern++;
        patternLen--;
        if (stringLen == 0) {
            while(*pattern == '*') {
                pattern++;
                patternLen--;
            }
            break;
        }
    }
    if (patternLen == 0 && stringLen == 0)
        return 1;
    return 0;
}

/* ========================== 'show' subcommand ============================= */

void gguf_tools_show(const char *filename) {
    gguf_ctx *ctx = gguf_open(filename);
    if (ctx == NULL) {
        perror("Opening GGUF file");
        exit(1);
    }

    /* Show general information about the neural network. */
    printf("%s (ver %d): %llu key-value pairs, %llu tensors\n",
        filename,
        (int)ctx->header->version,
        (unsigned long long)ctx->header->metadata_kv_count,
        (unsigned long long)ctx->header->tensor_count);

    /* Show all the key-value pairs. */
    gguf_key key;
    while (gguf_get_key(ctx,&key)) {
        printf("%.*s: [%s] ", (int)key.namelen, key.name, gguf_get_value_type_name(key.type));
        gguf_print_value(ctx,key.type,key.val,Opt.verbose);
        printf("\n");
    }

    /* Show all the tensors. */
    gguf_tensor tensor;
    uint64_t params = 0;
    while (gguf_get_tensor(ctx,&tensor)) {
        printf("%s tensor %.*s @%llu, %llu weights, dims ",
            gguf_get_tensor_type_name(tensor.type),
            (int)tensor.namelen,
            tensor.name,
            tensor.offset,
            tensor.num_weights);
        for (uint32_t j = 0; j < tensor.ndim; j++) {
            printf("%s%llu",(j == 0) ? "[" : ",", tensor.dim[j]);
        }
        printf("], %llu bytes\n", tensor.bsize);

        params += tensor.num_weights;
    }
    printf("gguf-tools.info.parameters: %.02fB\n",
        (double)params/1000000000);
    return;
}

/* ======================= 'split-mixtral' subcommand ======================= */

/* Read a Mixtral MoE model and creates a new non-MoE GGUF file based
 * on the weights of the experts with IDs in the array of 'experts_id'.
 * The array must contain 32 integers, one for each layer. */
void gguf_tools_split_mixtral(int *experts_id, const char *mixtral_filename, const char *output_filename) {
    gguf_ctx *mixtral = gguf_open(mixtral_filename);
    if (mixtral == NULL) {
        perror("Opening Mixtral file");
        exit(1);
    }

    gguf_ctx *output = gguf_create(output_filename, GGUF_NONE);
    if (output == NULL) {
        perror("Opening the output file");
        exit(1);
    }

    /* To start, copy all the key value items, excluding the one
     * related to the experts. */
    gguf_key key;
    while (gguf_get_key(mixtral,&key)) {
        char keybuf[1024];
        snprintf(keybuf,sizeof(keybuf),"%.*s",(int)key.namelen, key.name);

        int skip = strstr(keybuf,"llama.expert_") != NULL;

        if (!skip)
            printf("Copying %s\n", keybuf);
        uint64_t value_start_offset = mixtral->off;
        void *value = mixtral->data+mixtral->off;
        // Just consume the value without doing anything with it.
        gguf_do_with_value(mixtral,key.type,key.val,NULL,0,0,NULL);
        uint64_t value_len = mixtral->off - value_start_offset;

        // Now append the value to the output model.
        if (!skip)
            gguf_append_kv(output,key.name,key.namelen,key.type,value,value_len);
    }

    /* Now it's time to copy the tensors. We need to copy all the shared
     * tensors (between the different experts), but only a set of
     * expert-specific tensors corresponding to the expert ID the user
     * wants to extract. */
    struct tensor_to_copy {
        sds dest_name;          // Tensor name in the output file.
        gguf_tensor orig_info;  // Original tensor info.
        uint64_t dest_offset;   // Destination offset in output file.
        uint64_t size;          // Tensor total bytes.
    };

    uint32_t num_tensors = 0;
    uint32_t max_tensors = 2048;

    struct tensor_to_copy *tensors =
        malloc(sizeof(struct tensor_to_copy)*max_tensors);
    if (tensors == NULL) {
        perror("Allocating tensors info array");
        exit(1);
    }

    /* Scan Mixtral tensors looking for the ones we need to copy
     * in the output model. */
    gguf_tensor tensor_info;
    while (gguf_get_tensor(mixtral,&tensor_info)) {
        assert(num_tensors < max_tensors);

        char tn[1024]; // Tensor name as null terminated string.
        snprintf(tn,sizeof(tn),"%.*s",(int)tensor_info.namelen, tensor_info.name);

        /* The tensor is a feed-forward tensor? We want to copy only
         * the ones of our expert ID. */
        if (strstr(tn,".ffn_") != NULL && strstr(tn,".ffn_norm") == NULL) {
            /* Extract which block this FFN belongs. */
            int block;
            assert(memcmp(tn,"blk.",4) == 0); // Must start with blk.<block>
            char *p = strchr(tn+4,'.');
            assert(p != NULL);
            *p = 0;
            block = atoi(tn+4);
            *p = '.';
            assert(block >= 0 && block < 32);

            /* Now that we have the block, we can select the corresponding
             * expert ID we want to use for this block. */
            int expert_id = experts_id[block];

            char match[32];
            snprintf(match,sizeof(match),".%d.weight",expert_id);
            char *match_ptr = strstr(tn,match);
            if (match_ptr == NULL) {
                printf("Skipping tensor %s\n", tn);
                continue; // Skip this tensor.
            }

            /* We need to remove the .<id>. from the name. */
            size_t taillen = strlen(match_ptr);
            memmove(match_ptr,match_ptr+2,taillen+1);
        }

        /* Create the entry for this tensor. Later we will scan all our
         * entries and append data to our output tensor. */
        tensors[num_tensors].dest_name = sdsnew(tn);
        if (tensors[num_tensors].dest_name == NULL) {
            perror("Allocating test tensor name");
            exit(1);
        }
        tensors[num_tensors].orig_info = tensor_info;
        tensors[num_tensors].size = tensor_info.bsize;
        num_tensors++;
    }

    /* Now we need to set the offset for our destination tensors. As
     * we calculate the offsets, we can emit the tensors information
     * section as well. */
    uint64_t tensor_off = 0; // Tensor offsets are relative to data section,
                             // so we start at offset 0.
    for (uint32_t j = 0; j < num_tensors; j++) {
        /* Align offset. */
        tensor_off += gguf_get_alignment_padding(mixtral->alignment,tensor_off);
        tensors[j].dest_offset = tensor_off;
        if (gguf_append_tensor_info(output,tensors[j].dest_name,strlen(tensors[j].dest_name),tensors[j].orig_info.ndim,tensors[j].orig_info.dim,tensors[j].orig_info.type,tensor_off) == 0)
        {
            perror("Failed to append tensor info");
            exit(1);
        }
        tensor_off += tensors[j].orig_info.bsize;
    }
    printf("Output file: after writing tensors info, file size is: %llu\n", output->size);

    /* Finally, append the tensors weights. */
    for (uint32_t j = 0; j < num_tensors; j++) {
        printf("Writing tensor %s (weights from %.*s)\n", tensors[j].dest_name,
            (int)tensors[j].orig_info.namelen, tensors[j].orig_info.name);
        if (gguf_append_tensor_data(output,tensors[j].orig_info.weights_data,
            tensors[j].orig_info.bsize) == 0)
        {
            perror("Failed to append tensor data");
            exit(1);
        }
    }
    exit(0);
}

/* ====================== 'inspect-weights' subcommand ====================== */

void gguf_tools_inspect_weights(const char *filename, const char *tname, uint64_t count) {
    gguf_ctx *ctx = gguf_open(filename);
    if (ctx == NULL) {
        perror("Opening GGUF file");
        exit(1);
    }

    /* Skip all the key-value pairs. */
    gguf_skip_key_values_section(ctx);

    /* Look for the tensor with the specified name. */
    size_t tnamelen = strlen(tname);
    gguf_tensor tensor;
    while (gguf_get_tensor(ctx,&tensor)) {
        if (tensor.namelen != tnamelen ||
            memcmp(tensor.name,tname,tnamelen)) continue;
        break; // Matching tensor found!
    }

    if (tensor.name == NULL) {
        fprintf(stderr, "A tensor with the specified name was not found\n");
        exit(1);
    }

    float *weights = gguf_tensor_to_float(&tensor);
    if (weights == NULL) {
        if (errno == EINVAL) {
            fprintf(stderr,"Unsupported tensor type: %s\n",
                gguf_get_tensor_type_name(tensor.type));
        } else {
            fprintf(stderr,"Out of memory\n");
        }
        exit(1);
    }

    uint64_t strides[GGUF_TENSOR_MAX_DIM] = {0};
    strides[tensor.ndim-1] = 1;
    for (int j = tensor.ndim - 2; j >= 0; j--) {
        strides[j] = tensor.dim[tensor.ndim - 2 - j] * strides[j + 1];
    }

    const int ident = 4;
    uint64_t j = 0;
    int broke = 1;
    while (j < tensor.num_weights) {
        int last = j + 1 == tensor.num_weights;
        for (int k = 0; k < (int) tensor.ndim - 1; k++) {
            if (j % strides[k] == 0) {
                printf("%*s\n", k * ident, "[");
            }
        }
        if (broke) {
            printf("%*s", tensor.ndim * ident, "");
        }
        printf("%f%s", weights[j], last ? "" : ", ");
        broke = 0;
        j++;
        for (int k = (int) tensor.ndim - 2; k >= 0; k--) {
            if (j % strides[k] == 0) {
                if (!broke) {
                    broke = 1;
                    printf("\n");
                }
                printf("%*s%s\n", k * ident, "]", last ? "" : ",");
            }
        }
        if (!broke && j % 4 == 0) {
            broke = 1;
            printf("\n");
        }
        if (j == count) break;
    }
    if (!broke) printf("\n");
    free(weights);
    return;
}

/* ========================== 'compare' subcommand ========================== */

/* Given two tensors of the same length, return the average difference
 * of their weights, in percentage.
 *
 * The difference is calculated like that: the average of the absolute values
 * of all the weights in the two vectors is calculated. Then, for each set
 * of corresponding weights, we calculate the difference, and the percentage
 * according to the average value (100%). The function returns the average
 * of the percentage of difference between all the pairs.
 *
 * Returns 1 on success, 0 if one or both the provided tensors can't be
 * dequantized. */
int tensors_avg_diff(gguf_tensor *t1, gguf_tensor *t2, double *diff) {
    float *weights1 = gguf_tensor_to_float(t1);
    float *weights2 = gguf_tensor_to_float(t2);
    if (weights1 == NULL || weights2 == NULL) {
        if (weights1) free(weights1);
        if (weights2) free(weights2);
        return 0;
    }

    /* Compute the average magnitude of the weights. */
    double tot_mag = 0;
    for (uint64_t j = 0; j < t1->num_weights; j++) {
        tot_mag += fabs(weights1[j]);
        tot_mag += fabs(weights2[j]);
    }
    double avg_mag = tot_mag/(t1->num_weights*2);

    /* Compute the average % difference of the weights. */
    double tot_diff = 0;
    for (uint64_t j = 0; j < t1->num_weights; j++)
        tot_diff += fabs(weights1[j]-weights2[j]);
    double avg_diff = tot_diff / t1->num_weights;

    /* Multiply by 75 to normalize the difference of a
     * random varialbe between -N and +N to 0 - 100% */
    *diff = avg_diff / avg_mag * 75;

    free(weights1);
    free(weights2);
    return 1;
}

void gguf_tools_compare(const char *file1, const char *file2) {
    gguf_ctx *ctx1 = gguf_open(file1);
    gguf_ctx *ctx2 = gguf_open(file2);
    if (ctx1 == NULL || ctx2 == NULL) {
        perror("Opening GGUF files");
        exit(1);
    }

    /* Skip all the key-value pairs. */
    gguf_skip_key_values_section(ctx1);

    /* For each tensor of the first net... */
    gguf_tensor tensor1, tensor2;
    while (gguf_get_tensor(ctx1,&tensor1)) {
        gguf_skip_key_values_section(ctx2);
        while (gguf_get_tensor(ctx2,&tensor2)) {
            /* Search for a tensor with the same name. */
            if (tensor2.namelen == tensor1.namelen &&
                memcmp(tensor2.name,tensor1.name,tensor1.namelen) == 0)
            {
                printf("[%.*s]: ", (int)tensor1.namelen, tensor1.name);
                fflush(stdout);
                if (tensor1.num_weights != tensor2.num_weights) {
                    printf("size mismatch\n");
                } else {
                    double diff;
                    if (tensors_avg_diff(&tensor1, &tensor2, &diff)) {
                        printf("avg weights difference: %f%%\n", diff);
                    } else {
                        printf("dequantization function missing...\n");
                    }
                }
            }
        }
        gguf_rewind(ctx2);
    }
}

/* ======================= Main and CLI options parsing ===================== */

void gguf_tools_usage(const char *progname) {
    printf("Usage: %s <subcommand> [arguments...] [options...]\n"
"Subcommands:\n"
"  show <filename> -- show GGUF model keys and tensors.\n"
"  inspect-tensor <filename> <tensor-name> [count] -- show tensor weights.\n"
"  compare <file1> <file2> -- avg weights diff for matching tensor names.\n"
"  split-mixtral <ids...> mixtral.gguf out.gguf -- extract expert.\n"
"Options:\n"
"  --verbose       :With 'show', print full arrays (e.g. token lists)\n"
"Example:\n"
"  split-mixtral 65230776370407150546470161412165 mixtral.gguf out.gguf\n"
           , progname);
    exit(1);
}

int main(int argc, char **argv) {
    if (argc < 3) gguf_tools_usage(argv[0]);

    /* Parse options before getting into subcommands parsing. */
    for (int j = 1; j < argc; j++) {
        /* Every time we find a an option, we try to parse it
         * and set the used argv[] entires to NULL. Later we remove
         * the NULL entries. In this way '--options' can be anywhere,
         * making the tool simpler to use. */
        if (!strcmp(argv[j],"--verbose")) {
            argv[j] = NULL;
            argc--;
            Opt.verbose = 1;
        }
    }

    /* Strip empty elements. */
    for (int j = 1; j < argc; j++) {
        if (argv[j] == NULL) {
            memmove(argv+j, argv+j+1, sizeof(char*) * (argc-j));
        }
    }

    if (!strcmp(argv[1],"show") && argc == 3) {
        gguf_tools_show(argv[2]);
    } else if (!strcmp(argv[1],"compare") && argc == 4) {
        gguf_tools_compare(argv[2],argv[3]);
    } else if (!strcmp(argv[1],"inspect-tensor") && (argc == 4 || argc == 5)) {
        gguf_tools_inspect_weights(argv[2],argv[3],
                                   argc == 5 ? atoi(argv[4]) : 0);
    } else if (!strcmp(argv[1],"split-mixtral") && argc == 5) {
        int experts[32];
        size_t elen = strlen(argv[2]);
        for (size_t j = 0; j < 32; j++) {
            if (j < elen) {
                experts[j] = argv[2][j] - '0';
                if (experts[j] < 0 || experts[j] > 7) {
                    fprintf(stderr,"Invalid expert ID: %d\n", experts[j]);
                    exit(1);
                }
            } else {
                /* If there aren't 32 digits in the input, use the last
                 * one repeated up to the last layer. */
                experts[j] = j > 1 ? experts[j-1] : 0;
            }
        }
        gguf_tools_split_mixtral(experts,argv[3],argv[4]);
    } else {
        gguf_tools_usage(argv[0]);
    }
    return 0;
}
