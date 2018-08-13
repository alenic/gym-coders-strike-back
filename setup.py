import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(name='gym_coders_strike_back',
      version='0.0.1',
      install_requires=['gym'],
      packages=setuptools.find_packages(),
      url="https://github.com/alenic/gym-coders-strike-back",
)  