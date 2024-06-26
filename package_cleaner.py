# Original package list
package_list = """
alabaster                         0.7.12
appdirs                           1.4.4
asn1crypto                        1.5.1
atomicwrites                      1.4.0
attrs                             21.4.0
Babel                             2.10.1
backports.entry-points-selectable 1.1.1
backports.functools-lru-cache     1.6.4
bcrypt                            3.2.2
bitstring                         3.1.9
blist                             1.3.6
CacheControl                      0.12.11
cachy                             0.3.0
certifi                           2021.10.8
cffi                              1.15.0
chardet                           4.0.0
charset-normalizer                2.0.12
cleo                              0.8.1
click                             8.1.3
clikit                            0.6.2
colorama                          0.4.4
crashtest                         0.3.1
cryptography                      37.0.1
Cython                            0.29.28
decorator                         5.1.1
distlib                           0.3.4
docopt                            0.6.2
docutils                          0.17.1
ecdsa                             0.17.0
editables                         0.3
expecttest                        0.1.3
filelock                          3.6.0
flit                              3.7.1
flit_core                         3.7.1
fsspec                            2022.3.0
future                            0.18.2
glob2                             0.7
html5lib                          1.1
idna                              3.3
imagesize                         1.3.0
importlib-metadata                4.11.3
importlib-resources               5.7.1
iniconfig                         1.1.1
intervaltree                      3.1.0
intreehooks                       1.0
ipaddress                         1.0.23
jeepney                           0.8.0
Jinja2                            3.1.2
joblib                            1.1.0
jsonschema                        4.4.0
keyring                           23.5.0
keyrings.alt                      4.1.0
liac-arff                         2.5.0
lockfile                          0.12.2
MarkupSafe                        2.1.1
mock                              4.0.3
more-itertools                    8.12.0
msgpack                           1.0.3
netaddr                           0.8.0
netifaces                         0.11.0
packaging                         20.9
paramiko                          2.10.4
pastel                            0.2.1
pathlib2                          2.3.7.post1
pathspec                          0.9.0
pbr                               5.8.1
pexpect                           4.8.0
Pillow                            9.1.1
pip                               22.0.4
pkginfo                           1.8.2
platformdirs                      2.4.1
pluggy                            1.0.0
poetry                            1.1.13
poetry-core                       1.0.8
protobuf                          3.19.4
psutil                            5.9.0
ptyprocess                        0.7.0
py                                1.11.0
py-expression-eval                0.3.14
pyasn1                            0.4.8
pybind11                          2.9.2
pycparser                         2.21
pycrypto                          2.6.1
Pygments                          2.12.0
pylev                             1.4.0
PyNaCl                            1.5.0
pyparsing                         3.0.8
pyrsistent                        0.18.1
pytest                            7.1.2
python-dateutil                   2.8.2
pytoml                            0.1.21
pytz                              2022.1
PyYAML                            6.0
regex                             2022.4.24
requests                          2.27.1
requests-toolbelt                 0.9.1
scandir                           1.10.0
SecretStorage                     3.3.2
semantic-version                  2.9.0
setuptools                        62.1.0
setuptools-rust                   1.3.0
setuptools-scm                    6.4.2
shellingham                       1.4.0
simplegeneric                     0.8.1
simplejson                        3.17.6
six                               1.16.0
snowballstemmer                   2.2.0
sortedcontainers                  2.4.0
Sphinx                            4.5.0
sphinx-bootstrap-theme            0.8.1
sphinxcontrib-applehelp           1.0.2
sphinxcontrib-devhelp             1.0.2
sphinxcontrib-htmlhelp            2.0.0
sphinxcontrib-jsmath              1.0.1
sphinxcontrib-qthelp              1.0.3
sphinxcontrib-serializinghtml     1.1.5
sphinxcontrib-websupport          1.2.4
tabulate                          0.8.9
threadpoolctl                     3.1.0
toml                              0.10.2
tomli                             2.0.1
tomli_w                           1.0.0
tomlkit                           0.10.2
torch                             1.12.0
typing_extensions                 4.2.0
ujson                             5.2.0
urllib3                           1.26.9
virtualenv                        20.14.1
wcwidth                           0.2.5
webencodings                      0.5.1
wheel                             0.37.1
xlrd                              2.0.1
zipfile36                         0.1.3
zipp                              3.8.0

"""

# Split the text into lines
lines = package_list.strip().split('\n')

# Create a new list for cleaned package entries
cleaned_lines = []

# Process each line to format it correctly
for line in lines:
    if line.strip():  # Ignore empty lines
        package, version = line.rsplit(None, 1)
        cleaned_lines.append(f"{package}=={version}")

# Join the cleaned lines into a single string
cleaned_package_list = '\n'.join(cleaned_lines)

# Save to a requirements file
with open('cleaned_requirements.txt', 'w') as f:
    f.write(cleaned_package_list)

print("Cleaned requirements saved to cleaned_requirements.txt")