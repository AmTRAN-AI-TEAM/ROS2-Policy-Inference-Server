from setuptools import find_packages, setup

package_name = 'robomimic_policy_srv'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/models', ['models/model_epoch_20.pth']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='robertlo',
    maintainer_email='longann890621@gmail.com',
    description='Robomimic policy ROS2 service',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'policy_service_node = robomimic_policy_srv.policy_service_node:main',
        ],
    },
)
