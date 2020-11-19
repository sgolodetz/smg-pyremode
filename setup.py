import setuptools

# with open("README.md", "r") as fh:
#     long_description = fh.read()

setuptools.setup(
    name="smg-remode",
    version="0.0.1",
    author="Stuart Golodetz",
    author_email="stuart.golodetz@cs.ox.ac.uk",
    description="Python bindings for REMODE",
    long_description="",  #long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sgolodetz/smg-pyremode",
    packages=["smg.pyremode"],
    include_package_data=True,
    install_requires=[
        "numpy",
        "smg-open3d",
        "smg-opencv",
        "smg-openni",
        "smg-pyorbslam2",
        "smg-utility"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='==3.7.*',
)
