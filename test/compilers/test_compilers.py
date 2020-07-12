import argparse
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import pytest
from pytest import ExitCode
import shutil
import subprocess
import time

_NUMCPP_ROOT_DIR = (Path(__file__).parent / '..' / '..').resolve()
_PYTEST_DIR = _NUMCPP_ROOT_DIR / 'test' / 'pytest'
_UNIT_TEST_BIN_DIR = _NUMCPP_ROOT_DIR / 'test' / 'lib'
_EXAMPLE_DIR = _NUMCPP_ROOT_DIR / 'examples'
_BUILD_DIR_NAME = 'build'

_SUDO_REQUIRED = True
try:
    subprocess.check_call(['sudo', '-h'], stdout=subprocess.DEVNULL)
except Exception:
    _SUDO_REQUIRED = False


class SimpleTimer:
    def __init__(self, name: str = None):
        self._name = name
        self._start = time.perf_counter()

    def tic(self):
        self._start = time.perf_counter()

    def toc(self):
        print(f'{self._name} elapsed time: {time.perf_counter() - self._start:.1f} seconds')


class Compiler(Enum):
    GNU = 0
    Clang = 1


_COMPILERS = {Compiler.GNU: 'gcc',
              Compiler.Clang: 'clang'}

_COMPILER_VERSIONS = {Compiler.GNU: ['6', '7', '8', '9', '10'],
                      Compiler.Clang: ['6.0', '7', '8', '9', '10']}


class CxxStandard(Enum):
    cxx_14 = 0
    cxx_17 = 1
    cxx_20 = 2


_CXX_STANDARDS = {CxxStandard.cxx_14: '14',
                  CxxStandard.cxx_17: '17',
                  CxxStandard.cxx_20: '20'}


@dataclass
class BuildConfigs:
    boost_root: str = None
    eigen_root: str = None
    opencv_root: str = None
    cxx_standard: CxxStandard = None


class Target(Enum):
    NumCppPy = 0
    GaussNewtonNlls = 1
    InterfaceWithEigen = 2
    InterfaceWithOpenCV = 3
    ReadMe = 4


class Builder:
    def __init__(self, root_dir: str):
        self._root_dir = Path(root_dir)
        if not self._root_dir.is_dir():
            raise NotADirectoryError(f"Input 'root_dir' is not a valid directory.\n\t{root_dir}")
        self._build_dir = self._root_dir / _BUILD_DIR_NAME
        self._cmake_configured = False
        self._current_compiler = None
        self._current_compiler_version = None
        self.update_compiler(compiler=Compiler.GNU)
        self.update_compiler_version(version=_COMPILER_VERSIONS[Compiler.GNU][-1])

    def configure_cmake(self, build_configs: BuildConfigs = None):
        if self._build_dir.exists():
            shutil.rmtree(self._build_dir)

        self._build_dir.mkdir()

        cmake_cmd = ['cmake',
                     '-S', str(self._root_dir),
                     '-B', str(self._build_dir),
                     '-DNUMCPP_TEST=True',
                     '-DNUMCPP_EXAMPLES=True']

        if build_configs is not None:
            if build_configs.boost_root is not None:
                if not Path(build_configs.boost_root).exists():
                    raise NotADirectoryError(f'Input boost_root directory does not exist:\n\t'
                                             f'{build_configs.boost_root}')
                cmake_cmd.append(f'-DBOOSTROOT={build_configs.boost_root}')

            if build_configs.eigen_root is not None:
                if not Path(build_configs.eigen_root).exists():
                    raise NotADirectoryError(f'Input eigen_root directory does not exist:\n\t'
                                             f'{build_configs.eigen_root}')
                cmake_cmd.append(f'-DEigen3_DIR={build_configs.eigen_root}')

            if build_configs.opencv_root is not None:
                if not Path(build_configs.opencv_root).exists():
                    raise NotADirectoryError(f'Input opencv_root directory does not exist:\n\t'
                                             f'{build_configs.opencv_root}')
                cmake_cmd.append(f'-DOpenCV_DIR={build_configs.opencv_root}')

            if build_configs.cxx_standard is not None:
                cmake_cmd.append(f'-DCMAKE_CXX_STANDARD={_CXX_STANDARDS[build_configs.cxx_standard]}')

        subprocess.check_call(cmake_cmd)
        self._cmake_configured = True

    def build_target(self, target: Target):
        if not self._cmake_configured:
            self.configure_cmake()

        cmake_cmd = ['cmake',
                     '--build', str(self._build_dir),
                     '--config', 'Release',
                     '--target', target.name]
        subprocess.check_call(cmake_cmd)

    def build_all(self):
        self.build_unit_test()
        self.build_examples()

    def build_unit_test(self):
        self.delete_unit_test()
        self.build_target(target=Target.NumCppPy)
        self.check_unit_test_binary()

    def build_examples(self):
        self.build_gauss_newton_nlls()
        self.build_interface_with_eigen()
        self.build_interface_with_opencv()
        self.build_readme()

    def build_example_target(self, target: Target):
        self.delete_example(target=target)
        self.build_target(target=target)
        self.check_example_binary(target=target)

    def build_gauss_newton_nlls(self):
        self.build_example_target(target=Target.GaussNewtonNlls)

    def build_interface_with_eigen(self):
        self.build_example_target(target=Target.InterfaceWithEigen)

    def build_interface_with_opencv(self):
        self.build_example_target(target=Target.InterfaceWithOpenCV)

    def build_readme(self):
        self.build_example_target(target=Target.ReadMe)

    def update_compiler(self, compiler: Compiler):
        update_alternatives_cmd = ['update-alternatives',
                                   '--set', 'cc', f'/usr/bin/{_COMPILERS[compiler]}']
        if _SUDO_REQUIRED:
            update_alternatives_cmd.insert(0, 'sudo')

        subprocess.check_call(update_alternatives_cmd)
        self._current_compiler = compiler

    def update_compiler_version(self, version: str):
        if version not in _COMPILER_VERSIONS[self._current_compiler]:
            raise ValueError(f"Unknown version '{version}' for {self._current_compiler} compiler\n"
                             f'Valid options are {_COMPILER_VERSIONS[self._current_compiler]}')

        compiler_str = _COMPILERS[self._current_compiler]
        update_alternatives_cmd = ['update-alternatives',
                                   '--set', compiler_str, f'/usr/bin/{compiler_str}-{version}']
        if _SUDO_REQUIRED:
            update_alternatives_cmd.insert(0, 'sudo')

        subprocess.check_call(update_alternatives_cmd)
        self._current_compiler_version = version

    @staticmethod
    def delete_unit_test():
        binary = _UNIT_TEST_BIN_DIR / f'{Target.NumCppPy.name}.so'
        if binary.is_file():
            binary.unlink()

    @staticmethod
    def delete_example(target: Target):
        target_str = target.name
        output_file = _EXAMPLE_DIR / target_str / 'bin' / target_str
        if output_file.is_file():
            output_file.unlink()

    @staticmethod
    def check_unit_test_binary():
        binary = _UNIT_TEST_BIN_DIR / f'{Target.NumCppPy.name}.so'
        if not binary.is_file():
            raise RuntimeError(f'unit test not successfully built')

    @staticmethod
    def check_example_binary(target: Target):
        target_str = target.name
        binary = _EXAMPLE_DIR / target_str / 'bin' / target_str
        if not binary.is_file():
            raise RuntimeError(f'{target_str} target not successfully built')


def run_pytest(fileOrDirectory: str):
    exit_code = pytest.main([fileOrDirectory])
    if exit_code != ExitCode.OK:
        raise RuntimeError('Pytest failures')


def run_all(root_dir: str):
    timer = SimpleTimer(name='run_all')
    timer.tic()

    builder = Builder(root_dir=root_dir)

    build_configs = BuildConfigs()
    for compiler in Compiler:
        builder.update_compiler(compiler=compiler)

        for compiler_version in _COMPILER_VERSIONS[compiler]:
            builder.update_compiler_version(version=compiler_version)

            for cxx_standard in CxxStandard:
                build_configs.cxx_standard = cxx_standard
                builder.configure_cmake(build_configs=build_configs)
                builder.build_all()

                run_pytest(fileOrDirectory=str(_PYTEST_DIR))
                timer.toc()


def run_single(root_dir: str,
               compiler: Compiler,
               compiler_version: str,
               cxx_standard: CxxStandard,
               target: Target = None):
    timer = SimpleTimer(name=target.name)
    timer.tic()

    build_configs = BuildConfigs()
    build_configs.cxx_standard = cxx_standard

    builder = Builder(root_dir=root_dir)
    builder.update_compiler(compiler=compiler)
    builder.update_compiler_version(version=compiler_version)

    builder.configure_cmake(build_configs=build_configs)

    if target is None:
        builder.build_all()
    else:
        builder.build_target(target=target)

    timer.toc()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--compiler', type=str, required=False, default='GNU')
    parser.add_argument('-v', '--compiler_version', type=str, required=False, default='10')
    parser.add_argument('-s', '--cxx_standard', type=str, required=False, default='17')
    parser.add_argument('-t', '--target', type=str, required=False, default='all')
    parser.add_argument('-r', '--run_all', type=bool, required=False, default=False)
    args = parser.parse_args()

    if args.run_all:
        run_all(root_dir=_NUMCPP_ROOT_DIR)
    else:
        run_single(root_dir=_NUMCPP_ROOT_DIR,
                   compiler=Compiler[args.compiler],
                   compiler_version=args.compiler_version,
                   cxx_standard=CxxStandard[args.cxx_standard],
                   target=Target[args.target])
