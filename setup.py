from distutils.core import setup
setup(
  name = 'TORCHNDRAM',         # How you named your package folder (MyLib)
  packages = ['TORCHNDRAM'],   # Chose the same as "name"
  version = '0.1',      # Start with a small number and increase it with every change you make
  license='MIT',        # Chose a license from here: https://help.github.com/articles/licensing-a-repository
  description = 'PYTORCH IMPLEMENATION OF CHARTIER\'S AND PROULX\'S NDRAM',   # Give a short description about your library
  author = 'REY A. GONZALEZ',                   # Type in your name
  author_email = 'gonza736@purdue.edu',      # Type in your E-Mail
  url = 'https://github.com/ReyGunz/torchNDRAM',   # Provide either the link to your github or to your website
  download_url = 'https://github.com/ReyGunz/torchNDRAM/archive/refs/tags/init.tar.gz',
  keywords = ['NDRAM', 'ASSOCIATIVE', 'HEBBIAN'],   # Keywords that define your package best
  install_requires=[            # I get to this in a second
          'torch',
          'tqdm',
          'numpy'
      ],
  classifiers=[
    'Development Status :: 3 - Alpha',      # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
    'Intended Audience :: Developers',      # Define that your audience are developers
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',   # Again, pick a license
    'Programming Language :: Python :: 3',      #Specify which pyhton versions that you want to support
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
    'Programming Language :: Python :: 3.9',
    'Programming Language :: Python :: 3.10'
  ],
)
