# Loads python scripts in selected directory
# runs them and on completion moves them along with their output to a different directory

for python_script in queue/*.py
do
    filename=$(basename $python_script .py)
    PYTHONPATH=(realpath .):PYTHONPATH python $python_script > queue_done/${filename}_stdout 2> queue_done/${filename}_stderr
    mv $python_script queue_done/${filename}.py
done
