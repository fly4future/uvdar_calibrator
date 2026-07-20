from setuptools import setup

PACKAGE_NAME = "uvdar_calibrator"

setup(
    name=PACKAGE_NAME,
    version="1.0.0",
    packages=[
        PACKAGE_NAME,
        PACKAGE_NAME + ".engine",
        PACKAGE_NAME + ".diagnostics",
        PACKAGE_NAME + ".apps",
    ],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + PACKAGE_NAME]),
        ("share/" + PACKAGE_NAME, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    author="Vojtech Spurny",
    author_email="spurny@fly4future.com",
    maintainer="Vojtech Spurny",
    maintainer_email="spurny@fly4future.com",
    keywords=["ROS2"],
    classifiers=[
        "Intended Audience :: Developers",
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python",
        "Topic :: Software Development",
    ],
    description=(
        "OCamCalib-style omnidirectional camera calibration for UV LED grid "
        "targets, with a live cameracalibrator node mirroring ROS "
        "image_pipeline's camera_calibration."
    ),
    license="BSD-3-Clause",
    extras_require={
        'test': ['pytest'],
    },
    entry_points={
        "console_scripts": [
            "cameracalibrator = uvdar_calibrator.apps.live_node:main",
            "calibrate_offline = uvdar_calibrator.apps.cli:main",
        ],
    },
)
