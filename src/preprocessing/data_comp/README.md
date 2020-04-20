We should eventually delete this folder (data_comp)

Using this for now, to show-case the results of pre-processing in an easier way. It includes:

* `base_text` is the text (one line per paragraph per article per topic) produced from `topics = get_dataset_topics(DEV_TEST)` 
* `base_text_quotes_fixed` is `base_text` with *only* the spurious line breaks removed and quotation marks standarized
* `current_cleaned` is the current status with all preprocessing applied (ie commit 2f29e7316cd1a4129b8f9b463c3f1ccd100e1554) - I plan to update this as more preproccessing is added and will delete when no longer actively working on this piece 