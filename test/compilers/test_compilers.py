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

_TARGET_UNIT_TEST = 'NumCppPy'
_TARGET_GAUSS_NEWTON_NLLS = 'GaussNewtonNlls'
_TARGET_INTERFACE_WITH_EIGEN = 'InterfaceWithEigen'
_TARGET_INTERFACE_WITH_OPENCV = 'InterfaceWithOpenCV'
_TARGET_README = 'ReadMe'


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


_COMPILERS = {Compiler.GNU: 'g++',
              Compiler.Clang: 'clang++'}
_COMPILER_VERSIONS = {Compiler.GNU: [6, 7, 8, 9, 10],
                      Compiler.Clang: [8, 9, 10]}


class CxxStandard(Enum):
    cxx_14 = 0
    cxx_17 = 1
    cxx_20 = 2


_CXX_STANDARDS = {CxxStandard.cxx_14: '14',
                  CxxStandard.cxx_17: '17',
                  CxxStandard.cxx_20: '20'}


class Builder:
    @dataclass
    class BuildConfigs:
        boost_root: str = None
        eigen_root: str = None
        opencv_root: str = None
        compiler: Compiler = None
        cxx_standard: CxxStandard = None

    def __init__(self, root_dir: str):
        self._root_dir = Path(root_dir)
        self._build_dir = self._root_dir / _BUILD_DIR_NAME
        self._cmake_configured = False

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

            if build_configs.compiler is not None:
                cmake_cmd.append(f'-DCMAKE_CXX_COMPILER={_COMPILERS[build_configs.compiler]}')

            if build_configs.compiler is not None:
                cmake_cmd.append(f'-DCMAKE_CXX_STANDARD={_CXX_STANDARDS[build_configs.cxx_standard]}')

        subprocess.check_call(cmake_cmd)
        self._cmake_configured = True

    def build_target(self, target: str):
        if not self._cmake_configured:
            self.configure_cmake()

        cmake_cmd = ['cmake',
                     '--build', str(self._build_dir),
                     '--config', 'Release',
                     '--target', target]
        subprocess.check_call(cmake_cmd)

    def build_all(self):
        self.build_unit_test()
        self.build_examples()

    def build_unit_test(self):
        self.delete_unit_test()
        self.build_target(target=_TARGET_UNIT_TEST)
        self.check_unit_test_binary()

    def build_examples(self):
        self.build_gauss_newton_nlls()
        self.build_interface_with_eigen()
        self.build_interface_with_opencv()
        self.build_readme()

    def build_example_target(self, target: str):
        self.delete_example(target=target)
        self.build_target(target=target)
        self.check_example_binary(target=target)

    def build_gauss_newton_nlls(self):
        self.build_example_target(target=_TARGET_GAUSS_NEWTON_NLLS)

    def build_interface_with_eigen(self):
        self.build_example_target(target=_TARGET_INTERFACE_WITH_EIGEN)

    def build_interface_with_opencv(self):
        self.build_example_target(target=_TARGET_INTERFACE_WITH_OPENCV)

    def build_readme(self):
        self.build_example_target(target=_TARGET_README)

    @staticmethod
    def delete_unit_test():
        output_file = _UNIT_TEST_BIN_DIR / f'{_TARGET_UNIT_TEST}.so'
        if output_file.is_file():
            output_file.unlink()

    @staticmethod
    def delete_example(target: str):
        output_dir = _EXAMPLE_DIR / target / 'bin'
        output_file = output_dir / target
        if output_file.is_file():
            output_file.unlink()

    @staticmethod
    def check_unit_test_binary():
        binary = _UNIT_TEST_BIN_DIR / f'{_TARGET_UNIT_TEST}.so'
        if not binary.is_file():
            raise RuntimeError(f'unit test not successfully built')

    @staticmethod
    def check_example_binary(target: str):
        binary = _EXAMPLE_DIR / target / 'bin' / target
        if not binary.is_file():
            raise RuntimeError(f'{target} target not successfully built')

    @staticmethod
    def update_compiler_version(compiler: Compiler, version: int):
        compiler_str = _COMPILERS[compiler]
        update_alternatives_cmd = ['sudo', 'update-alternatives',
                                   '--set', compiler_str, f'/usr/bin/{compiler_str}-{version}']
        subprocess.check_call(update_alternatives_cmd)


def run_pytest(fileOrDirectory: str):
    exit_code = pytest.main([fileOrDirectory])
    if exit_code != ExitCode.OK:
        raise RuntimeError('Pytest failures')


def run_all(root_dir: str):
    timer = SimpleTimer(name='run_all')
    timer.tic()

    builder = Builder(root_dir=root_dir)

    build_configs = Builder.BuildConfigs()
    for i_compiler in range(len(Compiler)):
        build_configs.compiler = Compiler(i_compiler)

        for compiler_version in _COMPILER_VERSIONS[build_configs.compiler]:
            Builder.update_compiler_version(compiler=build_configs.compiler, version=compiler_version)

            for i_cxx_standard in range(len(CxxStandard)):
                build_configs.cxx_standard = CxxStandard(i_cxx_standard)
                builder.configure_cmake(build_configs=build_configs)
                builder.build_all()

                run_pytest(fileOrDirectory=str(_PYTEST_DIR))
                timer.toc()


if __name__ == '__main__':
    run_all(root_dir=_NUMCPP_ROOT_DIR)
