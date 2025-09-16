DeSTA2 implementation forked from [the official repository.](https://github.com/kehanlu/DeSTA2)

The original README is available at ```README_original.md```

We leverage DeSTA2 to answer the prompt "Using tone of voice only (prosody: pitch, rhythm, loudness, timbre). Ignore word meaning; do not transcribe. Reply with exactly one: angry | happy | sad | neutral" using emotionally incongruent audio samples from the Emotionally Incongruent Synthetic Speech dataset (EMIS).

The shell script ```run_desta2.sh``` automatically creates the conda environment used to evaluate this Spoken Language Model, following the method written on the paper. If you prefer, you can install packages via ```requirements.txt```.