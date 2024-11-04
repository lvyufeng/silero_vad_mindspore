rm -rf ./dist
python setup.py bdist_wheel
rm -rf *.egg-info
pip uninstall silero_vad_mindspore -y && pip install dist/*.whl