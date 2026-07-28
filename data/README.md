# data/

Raw datasets go here. They are git-ignored — never commit logs.

## Expected layout

```
data/
  hdfs/    # HDFS logs + labels from LogHub
  bgl/     # BGL logs + labels from LogHub
```

## Where to get them

LogHub: https://github.com/logpai/loghub  (HDFS, BGL)

## Week-1 note

The reference repo already preprocesses HDFS/BGL into template sequences. Reusing its
parsed output (rather than re-parsing raw logs) keeps our numbers comparable to theirs.
Wire `src/dataloader.py::_load_parsed_domain` to whichever source you use.
