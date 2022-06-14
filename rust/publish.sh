#!/bin/sh

HERE=`dirname $0`
cd "${HERE}"

if [ ! -d sppark ]; then
    trap '[ -h sppark ] && rm -f sppark' 0 2
    ln -s .. sppark
fi

# --allow-dirty because the temporary sppark symbolic link is not committed
cargo +stable publish --allow-dirty "$@"
