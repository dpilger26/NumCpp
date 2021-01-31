import argparse
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import platform
import shutil
import subprocess
import time
from typing import Union, List

_IS_WINDOWS = False
if platform.system() == 'Windows':
    _IS_WINDOWS = True

_DEFAULT_BOOST_DIR = None
_DEFAULT_EIGEN_DIR = None
_DEFAULT_OPENCV_DIR = None
if _IS_WINDOWS:
    _DEFAULT_BOOST_DIR = Path(r'C:\libs\boost\boost_1_73_0')
    _DEFAULT_BOOST_DIR_v141 = Path(r'C:\libs\boost\boost_1_68_0')
    _DEFAULT_EIGEN_DIR = Path(r'C:\Program Files (x86)\Eigen3\share\eigen3\cmake')
    _DEFAULT_OPENCV_DIR = Path(r'C:\libs\opencv\build')

_NUMCPP_ROOT_DIR = (Path(__file__).parent / '..' / '..').resolve()
_PYTEST_DIR = _NUMCPP_ROOT_DIR / 'test' / 'pytest'
_UNIT_TEST_LIB_DIR = _NUMCPP_ROOT_DIR / 'test' / 'lib'
_EXAMPLE_DIR = _NUMCPP_ROOT_DIR / 'examples'
_BUILD_DIR_NAME = 'build'
if _IS_WINDOWS:
    _BUILD_DIR_NAME += 'VS'

_SUDO_REQUIRED = True
try:
    subprocess.check_call(['sudo', '-h'], stdout=subprocess.DEVNULL)
except:  # noqa
    _SUDO_REQUIRED = False


class SimpleTimer:
    def __init__(self, name: str = None) -> None:
        self._name = name
        self._start = time.perf_counter()

    def tic(self):
        self._start = time.perf_counter()

    def toc(self):
        print(f'{self._name} elapsed time: {time.perf_counter() - self._start:.1f} seconds')


if _IS_WINDOWS:
    class Compiler(Enum):
        MSVC = 0

    _COMPILERS = {Compiler.MSVC: 'msbuild'}

    _COMPILER_VERSIONS = {Compiler.MSVC: ['v141', 'v142']}
else:
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

_CMAKE_DEFINES = ['NO_USE_BOOST', 'NO_MULTITHREAD']


@dataclass
class BuildConfigs:
    boost_root: Path = _DEFAULT_BOOST_DIR
    eigen_root: Path = _DEFAULT_EIGEN_DIR
    opencv_root: Path = _DEFAULT_OPENCV_DIR
    cxx_standard: CxxStandard = CxxStandard.cxx_17


class Target(Enum):
    NumCppPy = 0
    GaussNewtonNlls = 1
    InterfaceWithEigen = 2
    InterfaceWithOpenCV = 3
    ReadMe = 4
    all = 5


class Builder:
    def __init__(self, root_dir: str):
        self._root_dir = Path(root_dir)
        if not self._root_dir.is_dir():
            raise NotADirectoryError(f"Input 'root_dir' is not a valid directory.\n\t{root_dir}")
        self._build_dir = self._root_dir / _BUILD_DIR_NAME
        self._cmake_configured = False
        self._current_compiler = None
        self._current_compiler_version = None
        self._cmake_defines = []
        self.update_compiler(compiler=Compiler(0))
        self.update_compiler_version(version=_COMPILER_VERSIONS[Compiler(0)][-1])

    def configure_cmake(self, build_configs: BuildConfigs = None) -> None:
        if self._build_dir.exists():
            shutil.rmtree(self._build_dir)

        self._build_dir.mkdir()

        # NOTE: bypass InterfaceWithEigen example with VS on C++20 for now since Eigen doesn't
        # seem to compile. Revisit this in the future
        build_eigen_example = 'ON'
        if _IS_WINDOWS and build_configs.cxx_standard == CxxStandard.cxx_20:
            build_eigen_example = 'OFF'

        cmake_cmd = ['cmake',
                     '-S', str(self._root_dir),
                     '-B', str(self._build_dir),
                     '-DBUILD_TESTS=ON',
                     '-DBUILD_EXAMPLE_GAUSS_NEWTON_NLLS=ON',
                     f'-DBUILD_EXAMPLE_INTERFACE_WITH_EIGEN={build_eigen_example}',
                     '-DBUILD_EXAMPLE_INTERFACE_WITH_OPENCV=ON',
                     '-DBUILD_EXAMPLE_README=ON']

        for cmake_define in self._cmake_defines:
            cmake_cmd.append(f'-D{cmake_define}=True')

        if _IS_WINDOWS:
            cmake_cmd.extend(['-G', 'Visual Studio 16 2019',
                              '-A', 'x64',
                              f'-DCMAKE_GENERATOR_TOOLSET={self._current_compiler_version}'])

        if build_configs is not None:
            if build_configs.boost_root is not None:
                if not build_configs.boost_root.exists():
                    raise NotADirectoryError(f'Input boost_root directory does not exist:\n\t'
                                             f'{build_configs.boost_root}')
                cmake_cmd.append(f'-DBOOSTROOT={build_configs.boost_root}')

            if build_configs.eigen_root is not None and build_eigen_example:
                if not build_configs.eigen_root.exists():
                    raise NotADirectoryError(f'Input eigen_root directory does not exist:\n\t'
                                             f'{build_configs.eigen_root}')
                cmake_cmd.append(f'-DEigen3_DIR={build_configs.eigen_root}')

            if build_configs.opencv_root is not None:
                if not build_configs.opencv_root.exists():
                    raise NotADirectoryError(f'Input opencv_root directory does not exist:\n\t'
                                             f'{build_configs.opencv_root}')
                cmake_cmd.append(f'-DOpenCV_DIR={build_configs.opencv_root}')

            if build_configs.cxx_standard is not None:
                cmake_cmd.append(f'-DCMAKE_CXX_STANDARD={_CXX_STANDARDS[build_configs.cxx_standard]}')

        subprocess.check_call(cmake_cmd)
        self._cmake_configured = True

    def build_target(self, target: Target) -> None:
        if not self._cmake_configured:
            self.configure_cmake()

        cmake_cmd = ['cmake',
                     '--build', str(self._build_dir),
                     '--config', 'Release']
        if target != Target.all or not _IS_WINDOWS: 
            cmake_cmd.extend(['--target', target.name])
        subprocess.check_call(cmake_cmd)

    def build_all(self) -> None:
        self.build_target(Target.all)

    def build_unit_test(self) -> None:
        self.delete_unit_test()
        self.build_target(target=Target.NumCppPy)
        self.check_unit_test_binary()

    def build_examples(self) -> None:
        self.build_gauss_newton_nlls()
        self.build_interface_with_eigen()
        self.build_interface_with_opencv()
        self.build_readme()

    def build_example_target(self, target: Target) -> None:
        self.delete_example(target=target)
        self.build_target(target=target)
        self.check_example_binary(target=target)

    def build_gauss_newton_nlls(self) -> None:
        self.build_example_target(target=Target.GaussNewtonNlls)

    def build_interface_with_eigen(self) -> None:
        self.build_example_target(target=Target.InterfaceWithEigen)

    def build_interface_with_opencv(self) -> None:
        self.build_example_target(target=Target.InterfaceWithOpenCV)

    def build_readme(self) -> None:
        self.build_example_target(target=Target.ReadMe)

    def update_compiler(self, compiler: Compiler) -> None:
        if not _IS_WINDOWS:
            update_alternatives_cmd = ['update-alternatives',
                                       '--set', 'cc', f'/usr/bin/{_COMPILERS[compiler]}']
            if _SUDO_REQUIRED:
                update_alternatives_cmd.insert(0, 'sudo')

            subprocess.check_call(update_alternatives_cmd)

        self._current_compiler = compiler

    def update_compiler_version(self, version: str) -> None:
        if version not in _COMPILER_VERSIONS[self._current_compiler]:
            raise ValueError(f"Unknown version '{version}' for {self._current_compiler} compiler\n"
                             f'Valid options are {_COMPILER_VERSIONS[self._current_compiler]}')

        if not _IS_WINDOWS:
            compiler_str = _COMPILERS[self._current_compiler]
            update_alternatives_cmd = ['update-alternatives',
                                       '--set', compiler_str, f'/usr/bin/{compiler_str}-{version}']
            if _SUDO_REQUIRED:
                update_alternatives_cmd.insert(0, 'sudo')

            subprocess.check_call(update_alternatives_cmd)

        self._current_compiler_version = version

    def add_cmake_defines(self, defines: Union[List[str], str]) -> None:
        if type(defines) is str:
            defines = [defines]
        self._cmake_defines = defines

    @staticmethod
    def delete_unit_test() -> None:
        binary = _UNIT_TEST_LIB_DIR / f'{Target.NumCppPy.name}'
        if _IS_WINDOWS:
            binary = binary.with_suffix('.pyd')
        else:
            binary = binary.with_suffix('.so')
        if binary.is_file():
            binary.unlink()

    @staticmethod
    def delete_example(target: Target) -> None:
        target_str = target.name
        binary = _EXAMPLE_DIR / target_str / 'bin' / target_str
        if _IS_WINDOWS:
            binary = binary.with_suffix('.exe')
        if binary.is_file():
            binary.unlink()

    @staticmethod
    def check_unit_test_binary() -> None:
        binary = _UNIT_TEST_LIB_DIR / f'{Target.NumCppPy.name}'
        if _IS_WINDOWS:
            binary = binary.with_suffix('.pyd')
        else:
            binary = binary.with_suffix('.so')
        if not binary.is_file():
            raise RuntimeError(f'unit test not successfully built:\n\t{binary}')

    @staticmethod
    def check_example_binary(target: Target) -> None:
        target_str = target.name
        binary = _EXAMPLE_DIR / target_str / 'bin' / target_str
        if _IS_WINDOWS:
            binary = binary.with_suffix('.exe')
        if not binary.is_file():
            raise RuntimeError(f'{target_str} target not successfully built:\n\t{binary}')


def run_pytest(fileOrDirectory: str) -> None:
    # spawn a seperate process to avoid this python instance from owning the dll/so
    subprocess.check_call(['pytest', fileOrDirectory])


def run_single(root_dir: str,
               compiler: Compiler,
               compiler_version: str,
               cxx_standard: CxxStandard,
               no_use_boost: bool = False,
               no_multithread: bool = False,
               target: Target = None) -> None:
    print('=====================================================================================')
    print(f'root_dir:         {root_dir}')
    print(f'compiler:         {compiler}')
    print(f'compiler_version: {compiler_version}')
    print(f'cxx_standard:     {cxx_standard}')
    print(f'no_use_boost:     {no_use_boost}')
    print(f'no_multithread:   {no_multithread}')
    print('=====================================================================================')

    timer = SimpleTimer(name=target.name if target is not None else 'All')
    timer.tic()

    build_configs = BuildConfigs()
    build_configs.cxx_standard = cxx_standard
    if _IS_WINDOWS and compiler_version == 'v141':
        build_configs.boost_root = _DEFAULT_BOOST_DIR_v141

    builder = Builder(root_dir=root_dir)
    builder.update_compiler(compiler=compiler)
    builder.update_compiler_version(version=compiler_version)

    cmake_defines = []
    if no_use_boost:
        cmake_defines.append('NO_USE_BOOST')
    if no_multithread:
        cmake_defines.append('NO_MULTITHREAD')
    builder.add_cmake_defines(defines=cmake_defines)

    builder.configure_cmake(build_configs=build_configs)

    if target is None:
        builder.build_all()
    else:
        builder.build_target(target=target)

    if (target == Target.NumCppPy or 
            target == Target.all):
        run_pytest(fileOrDirectory=str(_PYTEST_DIR))

    timer.toc()


def run_all(root_dir: str) -> None:
    timer = SimpleTimer(name='run_all')
    timer.tic()

    for compiler in Compiler:
        for compiler_version in _COMPILER_VERSIONS[compiler]:
            for cxx_standard in CxxStandard:
                for cmake_define in [*_CMAKE_DEFINES, _CMAKE_DEFINES, []]:
                    run_single(root_dir=root_dir,
                               compiler=compiler,
                               compiler_version=compiler_version,
                               cxx_standard=cxx_standard,
                               no_use_boost='NO_USE_BOOST' in cmake_define,
                               no_multithread='NO_MULTITHREAD' in cmake_define)
                    run_pytest(fileOrDirectory=str(_PYTEST_DIR))
    timer.toc()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    if _IS_WINDOWS:
        default_compiler = 'MSVC'
        default_compiler_version = 'v142'
    else:
        default_compiler = 'GNU'
        default_compiler_version = '10'

    parser.add_argument('-c', '--compiler', type=str, required=False, default=default_compiler)
    parser.add_argument('-v', '--compiler_version', type=str, required=False, default=default_compiler_version)
    parser.add_argument('-s', '--cxx_standard', type=str, required=False, default='cxx_17')
    parser.add_argument('-t', '--target', type=str, required=False, default='all')
    parser.add_argument('-nb', '--no_use_boost', dest='no_use_boost', action='store_true', required=False)
    parser.add_argument('-nmt', '--no_multithread', dest='no_multithread', action='store_true', required=False)
    parser.add_argument('-r', '--run_all', dest='run_all', action='store_true', required=False)
    args = parser.parse_args()

    if args.run_all:
        run_all(root_dir=_NUMCPP_ROOT_DIR)
    else:
        run_single(root_dir=_NUMCPP_ROOT_DIR,
                   compiler=Compiler[args.compiler],
                   compiler_version=args.compiler_version,
                   cxx_standard=CxxStandard[args.cxx_standard],
                   no_use_boost=args.no_use_boost,
                   no_multithread=args.no_multithread,
                   target=Target[args.target])
