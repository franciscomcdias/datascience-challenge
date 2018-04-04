from setuptools import setup
from setuptools.command.install import install as _install


class Install(_install):
    def run(self):
        _install.do_egg_install(self)
        import nltk
        nltk.download("averaged_perceptron_tagger")
        nltk.download("punkt")


setup(
    name="datascience-challenge",
    version="1.0",
    description="datascience-challenge",
    author="Francisco Dias",
    author_email="mail@franciscodias.pt",
    cmdclass={"install": Install},
    install_requires=["nltk", ],
    setup_requires=["nltk"]
)
