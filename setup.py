from distutils.core import setup

setup(
    name='chaosgame',  # How you named your package folder (MyLib)
    packages=['chaosgame'],  # Chose the same as "name"
    version='0.1',  # Start with a small number and increase it with every change you make
    license='MIT',  # Chose a license from here: https://help.github.com/articles/licensing-a-repository
    description='A chaos game fractal generation library.',  # Give a short description about your library
    author='Rinze Watanabe',  # Type in your name
    author_email='rinze.watanabe.translation@gmail.com',  # Type in your E-Mail
    url='https://github.com/RW21/chaosgame',  # Provide either the link to your github or to your website
    download_url='https://github.com/RW21/chaosgame/archive/v0.1-beta.tar.gzhttps://github.com/RW21/chaosgame/archive/v0.1-beta.tar.gz',
    # I explain this later on
    keywords=['chaos game'],  # Keywords that define your package best
    install_requires=[
        'numpy',
        'matplotlib',
        'mpl_toolkits',
        'scipy',
        'shapely',
        'ipyvolume'
    ],
    classifiers=[
        'Development Status :: 4 - Beta',
        # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
        'Intended Audience :: Developers',  # Define that your audience are developers
        'Topic :: Software Development :: Build Tools',
        'License :: OSI Approved :: MIT License',  # Again, pick a license
        'Programming Language :: Python :: 3'  # Specify which pyhton versions that you want to support
    ]
)
