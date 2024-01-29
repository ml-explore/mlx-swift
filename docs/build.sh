#/bin/zsh

# based on https://github.com/swiftviz/Scale/blob/main/docbuild.bash

if [[ `git rev-parse --abbrev-ref HEAD` != 'gh-pages' ]]
then
    if [[ $1 != "-f" ]] 
    then
        echo "Documentation should only be build on the gh-pages branch.  Use -f to force build."
        exit 1
    fi
fi

cd ..

export DOCC_JSON_PRETTYPRINT=YES

for x in MLX MLXRandom MLXNN; do
    swift package \
	--allow-writing-to-directory ./docs \
	generate-documentation \
	--fallback-bundle-identifier mlx.swift.`echo $x | tr A-Z a-z` \
	--target $x \
	--output-path ./docs/build/$x \
	--emit-digest \
	--disable-indexing \
	--transform-for-static-hosting \
	--hosting-base-path mlx-swift/$x \
	--source-service github \
	--source-service-base-url https://github.com/ml-explore/mlx-swift/blob/main \
	--checkout-path `pwd`
done
