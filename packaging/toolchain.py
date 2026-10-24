"""Relocate the pinned HQ image's GCC/binutils/sysroot for CPU and CUDA JIT."""
from pathlib import Path
import os
import shutil


def assemble(out):
    out = Path(out)
    tree = out / 'toolchain'
    for source, destination in (('/usr/include', 'usr/include'), ('/usr/lib/gcc', 'usr/lib/gcc'),
                                ('/usr/libexec/gcc', 'usr/libexec/gcc'),
                                ('/usr/lib/x86_64-linux-gnu', 'usr/lib/x86_64-linux-gnu')):
        shutil.copytree(source, tree / destination, symlinks=True)
    (tree / 'lib').symlink_to('usr/lib')
    (tree / 'lib64').mkdir()
    # libc.so's linker script names this absolute loader inside the sysroot.
    shutil.copy2(Path('/lib64/ld-linux-x86-64.so.2').resolve(), tree / 'lib64/ld-linux-x86-64.so.2')
    binaries = tree / 'usr/bin'
    binaries.mkdir(parents=True)
    for source in Path('/usr/bin').glob('x86_64-linux-gnu-*'):
        if source.is_file():
            shutil.copy2(source.resolve(), binaries / source.name)
    for name in ('gcc', 'g++', 'as', 'ld', 'ar', 'nm', 'objcopy', 'objdump', 'ranlib', 'readelf', 'strip'):
        shutil.copy2(Path('/usr/bin', name).resolve(), binaries / name)
    # Executables run against the host's compatible glibc, not a competing libc.
    lib = out / 'lib'
    lib.mkdir(exist_ok=True)
    for pattern in ('libisl.so*', 'libmpc.so*', 'libmpfr.so*', 'libgmp.so*', 'libzstd.so*',
                    'libbfd-*.so', 'libctf.so*', 'libjansson.so*', 'libsframe.so*', 'libz.so*'):
        for source in Path('/usr/lib/x86_64-linux-gnu').glob(pattern):
            shutil.copy2(source.resolve(), lib / source.name)
    notices = out / 'licenses/toolchain'
    notices.mkdir(parents=True)
    for name in ('gcc-13', 'g++-13', 'cpp-13', 'gcc-13-base', 'binutils', 'binutils-common',
                 'libc6', 'libc6-dev', 'linux-libc-dev', 'libstdc++-13-dev', 'libgcc-13-dev',
                 'libgmp10', 'libmpc3', 'libmpfr6', 'libisl23', 'libzstd1', 'libsframe1',
                 'libctf0', 'libbinutils', 'libjansson4'):
        copyright_file = Path('/usr/share/doc') / name / 'copyright'
        if copyright_file.is_file():
            shutil.copy2(copyright_file.resolve(), notices / (name + '.copyright'))
    wrappers = tree / 'bin'
    wrappers.mkdir()
    for alias, binary in (('gcc', 'gcc'), ('cc', 'gcc'), ('g++', 'g++'), ('c++', 'g++')):
        script = wrappers / alias
        script.write_text('''#!/bin/sh
set -eu
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
export LD_LIBRARY_PATH="$ROOT/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
exec "$ROOT/toolchain/usr/bin/''' + binary + '''" \\
  -B"$ROOT/toolchain/usr/libexec/gcc/x86_64-linux-gnu/13/" \\
  -B"$ROOT/toolchain/usr/lib/gcc/x86_64-linux-gnu/13/" \\
  -B"$ROOT/toolchain/usr/bin/" --sysroot="$ROOT/toolchain" "$@"
''')
        os.chmod(script, 0o755)


if __name__ == '__main__':
    import sys
    assemble(sys.argv[1])
