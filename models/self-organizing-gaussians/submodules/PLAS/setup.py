from setuptools import setup, find_packages

setup(
    name='plas_sogg',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'torch',
        'kornia',
        'tqdm',

        # example/eval deps, maybe split?
        # sort_3d_gaussians
        'pandas',
        'plyfile',
        'trimesh',

        # sort_rgb_img
        'click',
        'pillow',


        # eval/compare_plas_flas
        'opencv-python',

        # eval/flas
        'lapx', # Originally had `lap` but `lap` was broken! Switched to `lapx`: https://pypi.org/project/lapx/, see https://github.com/gatagat/lap/issues/45
        'matplotlib',
        'scipy'
    ],
    entry_points={
        'console_scripts': [
        ],
    },
    package_data={
        'plas': ['../img/*.jpg'],
    },
    include_package_data=True,
)
