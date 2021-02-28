rm -rf dist/*
python setup.py sdist bdist_wheel
pip install certifi --upgrade
twine upload dist/*