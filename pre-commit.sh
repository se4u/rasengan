# INSTALLATION: ln -s pre-commit.sh .git/hooks/pre-commit
# USAGE: git commit

# Don't test code that won't be committed.
git stash -q --keep-index

# Check that we can import rasengan on python[23]
source activate py35
python -c 'import rasengan' || exit 1
source deactivate
python -c 'import rasengan' || exit 1

# Restore the stash
git stash pop -q
