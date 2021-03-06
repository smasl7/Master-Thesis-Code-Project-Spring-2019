# Master Thesis Work in Spring 2019 at KTH - The Code Project
This Github repository contains the code project that I have coded as part of my master thesis work _"Attribute-Driven Generation of Drug Reviews using Deep Learning"_ to produce all the relevant results. The entire code project is to be considered as an extension of the master thesis report that is included in this Github repository as _report.pdf_.

To run the experiments to produce the (roughly equivalent) results that were later published in the master thesis report, then it is enough to merely run **python main.py** (the main file is located in the _Programming_ folder) in a CLI window or simply run the _main.py_ file in an IDE that supports execution of Python scripts. Note, however, that not all necessary files are available in this Github repository due to memory limitations that are imposed on Github repositories. To be able to run the implementation, one must 1) download the GloVe word embeddings, and 2) download the frequency dictionary that is used for correcting drug reviews during preprocessing phase. To acquire these components, please read the "Appendices" section that is located at the end of the master thesis report (i.e. _report.pdf_) and that also provides a more detailed explanation to how to acquire those specified components.

If there are some issues or some further questions about details not being clarified, then feel free to contact me by through my email account smasl94@yahoo.se and I will try to answer as best as possible from my position. It is personally recommended to read the meaning of the MIT license that is attached along with this Github repository.

Have fun using this code project!

/Sylwester Liljegren

NOTE: Toward the end of the master thesis work, the codes for the V-Att2Seq model (as specified in the file _v_att2seq_model.py_) were accidentally deleted and therefore, the current version is a reconstruction of the same codes using the report as a guideline. The reason for mentioning this detail is to make the user aware that not exactly the same results may be reproduced using the reconstruction of the original codes. However, as some low-scaled experiments that were run in prior to the upload of these codes have shown, the codes should work as equally as the original codes.
