# selectify

Instal dependencies:

```
conda create selectify
conda activate selectify

pip install numpy matplotlib ont-fast5-api read-until toml
pip install jsonschema mappy npy-append-array ont-pyguppy-client-lib

pip install cudatoolkit==11.2.2 cudnn==8.1.0 keras==2.11 tensorflow==2.11
```

The version of `ont-pyguppy-client-lib` must match the version of your Guppy installation.
The tensorflow installation is compatible with CUDA 11.2.

To run the sequencing emulation, you must have stored stored sequencing run in `fast5` format available.

First, generate emulation index:

```
python3 run_simulation/read_indexer.py --fast5-reads ./reads/--output-dir ./index/ 
```

Run emulation without adaptive sampling using `test_app`:

```
python3 application/test_app.py
```

The paths to fast5 read folder and index folder need to be configured manually in the code.

You can emulate selective sequencing using one of integrated adaptive sampling applications. For now you can choose between `Readfish` and `selectify`.

To run Readfish, please study [Readfish guide](https://github.com/LooseLab/readfish). However, if you happen to possess the sequencing run produced by SARS-CoV-2 sample or ZymoBIOMICS standard sample, our example configuration should do just fine.

Run emulation using `Readfish`:

```
python3 integration/readfish/readfish.py targets --fast5-reads ./zymo/ --sorted-reads ./index/ --experiment-name exp1 --toml ./integration/readfish/config/saccharomyces_cerevisiae_negative_cfg.toml --log-file integration/readfish/log/targets.log
```

Run emulation using `selectify` (only SARS-CoV-2 is supported):

```
python3 integration/selectify/selectify.py --fast5-reads ./covid/ --sorted-reads ./index/ --sequencing-output ./out/emulation.bin --model integration/selectify/models/model/ --chunk-length 2000 --split-read-interval 0.55
```

To evalute the sequencing run, you need to base call the raw sequencing data in `fast5` files producing `fastq` files. Base called reads need to be mapped to the reference sequence using minimap2 producing a `paf` file.

Then evaluate data:

```
python3 statistics_evaluation/selection_evaluator.py --sequencing-output ./out/emulation.bin --minimap-output ./minimap_covid/output.paf
```

If you have any questions, you can contact me at <matej.fedor.mf@gmail.com>.