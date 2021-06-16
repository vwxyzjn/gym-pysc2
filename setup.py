from setuptools import setup
import versioneer

setup(name='gym_pysc2',
      version=versioneer.get_version(),
      cmdclass=versioneer.get_cmdclass(),
      install_requires=['gym']  # And any other dependencies foo needs
)