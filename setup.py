from distutils.core import setup
setup(
  name = 'QuickSound',         
  packages = ['QuickSound'],   
  version = '0.5',      
  license='MIT',       
  description = 'QuickSound allows for rapid and simple generation of sample sounds at any samplerate.',   
  author = 'Antonin Verdier',                   
  author_email = 'antonin@verdier.fr',    
  url = 'https://github.com/Pouple/QuickSound',   
  download_url = 'https://github.com/Pouple/QuickSound/archive/refs/tags/0.3.tar.gz',
  keywords = ['Sound', 'Signal', 'Generation', 'Auditory', 'Music', 'Simple', 'Small'],   
  install_requires=[            
          'numpy',
          'scipy',
          'sklearn'
      ],
  classifiers=[
    'Development Status :: 3 - Alpha',      
    'Intended Audience :: Developers',     
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',   
    'Programming Language :: Python :: 3',      
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
    'Programming Language :: Python :: 3.9',
  ],
)

