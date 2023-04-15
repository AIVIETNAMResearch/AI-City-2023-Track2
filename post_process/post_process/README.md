# Post-Process Module

During our cross-modal model, we don't explicitly add fine-grain constraints on the cross-modal similarity learning process, so we propose to add fine-grain constraints by post-process. Our post-process mainly consider two constraints, track attribute constraints and scene constraints. 
This module will do the following tasks:
- According the target tracks's attributes(color,type, and direction) constraints and related tracks's attributes(front/back/nearby track's color and type) constraints to refine the retrieval results.
- According some particular scene constraints(s-bent, parking spot,etc.) to refine the retrieval results

## Module organization 
- `relation-detect`   : Get the confusion matrix of the related vehicles for target tracks.
- `post-process-part1`: Post-process about the track attribute constraints.
- `post-process-part2`: Post-process about the scene constraints.

## Related Vehicles Detection
In order to refine the original retrieval result, we proposed to use the information of the vehicles which is related with the target vehicle, so we first detect the related vehicles for each tracked vehicle. And then obtain their color and type labels to generate the confusion matrix. See [relation-det](./relation-detect/README.md) for more details.

## Run Post-process
First, you need make sure the similarity matrix's path in the 'post-process-part1/post_process.py' is correct, and then run
```
cd post-process-part1
sh run.sh
```
After running this command, then go to the 'post-process-part2/' folder, and make sure the path in 'post_process.py' is correct, and then, run the following command:
```
sh run.sh
```
The final submission json file will be saved in the 'results/' folder.

