#!/bin/zsh

# See MAINTENANCE.md : Updating `mlx` and `mlx-c`

set -e

if [[ ! -d Source ]]
then
    echo "Please run from the root of the repository, e.g. ./tools/update-mlx.sh"
    exit 1
fi

# prepare public headers for Cmlx
rm -f Source/Cmlx/include-framework/*.h
cp Source/Cmlx/mlx-c/mlx/c/*.h Source/Cmlx/include-framework

# rewrite paths
for x in Source/Cmlx/include-framework/*.h ; do \
    sed -i .tmp -e 's:"mlx/c/:<Cmlx/mlx-c-:g' -e 's:#include ":#include <Cmlx/mlx-c-:g' -e 's:.h":.h>:g' $x
done;
rm Source/Cmlx/include-framework/*.tmp

for x in Source/Cmlx/include-framework/*.h ; do \
	mv $x `echo $x | sed -e 's:/include-framework/:/include-framework/mlx-c-:g'`
done;

# build the top level header
cat > Source/Cmlx/include-framework/Cmlx.h <<EOF
#include <Cmlx/mlx-c-mlx.h>
#include <Cmlx/mlx-c-transforms_impl.h>
#include <Cmlx/mlx-c-linalg.h>
#include <Cmlx/mlx-c-fast.h>

EOF

# c++ headers for xcodeproj -- these are transitively reachable
# from mlx/mlx/mlx.h
for x in \
    api.h \
    array.h \
    backend/cuda/cuda.h \
    backend/metal/metal.h \
    compile.h \
    device.h \
    distributed/distributed.h \
    distributed/ops.h \
    einsum.h \
    export.h \
    fast.h \
    fft.h \
    io.h \
    linalg.h \
    memory.h \
    ops.h \
    random.h \
    stream.h \
    transforms.h \
    utils.h \
    version.h \
    allocator.h \
    dtype.h \
    event.h \
    small_vector.h \
    types/complex.h \
    types/half_types.h \
    types/bf16.h \
    types/fp16.h \
    io/load.h \
    export_impl.h \
    threadpool.h \
    scheduler.h \
    primitives.h \
    backend/metal/device.h \
    backend/metal/utils.h \
    backend/common/utils.h \
    backend/cpu/encoder.h \
    backend/gpu/eval.h
do
    # guard the contents for non c++ callers
    h=mlx-`echo $x | tr / -`
    d=Source/Cmlx/include-framework/$h
    echo "#ifdef __cplusplus" > $d
    cat Source/Cmlx/mlx/mlx/$x | sed -e 's:backend/:backend-:g' -e 's:cuda/:cuda-:g' -e 's:gpu/:gpu-:g' -e 's:metal/:metal-:g' -e 's:distributed/:distributed-:g' -e 's:types/:types-:' -e 's:io/:io-:' -e 's:common/:common-:' -e 's:cpu/:cpu-:' -e 's:#include "mlx/:#include <Cmlx/mlx-:g' -e 's:#include ":#include <Cmlx/mlx-:g' -e 's:.h":.h>:g' -e 's:Metal/Metal.hpp:Cmlx/Metal.hpp:g' >> $d
    echo "#endif" >> $d

    # add to Cmlx
    echo "#include <Cmlx/$h>" >> Source/Cmlx/include-framework/Cmlx.h
done

# build & copy in the Metal.hpp header
(cd Source/Cmlx/metal-cpp; ./SingleHeader/MakeSingleHeader.py -o ../include-framework/Metal.hpp.in Foundation/Foundation.hpp QuartzCore/QuartzCore.hpp Metal/Metal.hpp MetalFX/MetalFX.hpp)

echo "#ifdef __cplusplus" > Source/Cmlx/include-framework/Metal.hpp
cat Source/Cmlx/include-framework/Metal.hpp.in >> Source/Cmlx/include-framework/Metal.hpp
echo "#endif" >> Source/Cmlx/include-framework/Metal.hpp
rm Source/Cmlx/include-framework/Metal.hpp.in

echo "#include <Cmlx/Metal.hpp>" >> Source/Cmlx/include-framework/Cmlx.h
