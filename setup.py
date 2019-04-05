import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
	name='psytrack',
	version='1.0.0',
	description='Tool for tracking dynamic psychometric curves',
	long_description=long_description,
	long_description_content_type="text/markdown",
	url='http://github.com/nicholas-roy/psytrack',
	author='Nicholas A. Roy, Ji Hyun Bak, and Jonathan W. Pillow',
	author_email='nicholas.roy.42@gmail.com',
	license='MIT',
    packages=setuptools.find_packages(),
	install_requires=[
          'numpy',
		  'scipy',
		  'matplotlib',
		  'markdown',
      ],
	classifiers=[
		"Programming Language :: Python :: 3",
		"License :: OSI Approved :: MIT License",
		"Operating System :: OS Independent",
		"Intended Audience :: Science/Research",
		"Topic :: Scientific/Engineering",
		],
	zip_safe=False,
	include_package_data=True)