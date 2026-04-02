
## Code repository for the paper 'Three Data Bipartitioning Methods to Improve Prediction of Time to Death'

<a href="https://github.com/KitDarkfeather/survival-bipartition">Three Data Bipartitioning Methods to Improve Prediction of Time to Death</a> ©2026 by <a href="https://kitdarkfeather.github.io/index.html">Michael Sossi</a> is licensed under <a href="https://creativecommons.org/licenses/by-sa/4.0/">Creative Commons Attribution-ShareAlike 4.0 International</a> <img src="https://mirrors.creativecommons.org/presskit/icons/cc.svg" alt="" style="width:20px;"><img src="https://mirrors.creativecommons.org/presskit/icons/by.svg" alt="" style="width:20px;"><img src="https://mirrors.creativecommons.org/presskit/icons/sa.svg" alt="" style="width:20px;">

- _In order to run the partitioning and modelling code, you will need access to the ELSA dataset.  Please contact the first author for instructions for gaining access or with any questions._  
- _Once ELSA data access is granted, the full **elsa.py** file will be provided._
- _The version of **elsa.py** in this repository has been stripped of ELSA metadata._

### umberto package
- [ ] Holds the data and datasets.

    #### library folder
    - Once permission is obtained, the ELSA raw data files can be stored in one of the the **umberto.library.ELSA** folders (e.g., the **tab** folder for *.tab files).
    - The freely available Veteran dataset is available in the **survset** folder.

    #### datasets package
    - The **elsa.py** file containing the full ELSA class will be provided after ELSA data permission is obtained.
    - Running **ELSA.build()** will build and store the ELSA dataset from the raw files.

### sapient package
- [ ] Contains the plotting, segmentation, and modelling code.

    #### plots package

    - The **plots.py** file contains the **Plots** parent class and associated functions for plotting survival analysis curves.
    - The **KaplanMeier** class is used to plot Kaplan-Meier curves.
    - The **Brier** class is used to plot Brier score curves.
    - The Veteran dataset can be used to test the Kaplan-Meier curve plotting:
    ```python
  from sapient.plots.kaplan_meier import KaplanMeier
      
  KaplanMeier('veteran').plot('kaplan-meier', 'days', size=22, width=5)
    ```
  
    #### segment package

    - The **segment.py** file contains the **Segment** class and associated functions for creating and evaluating bipartitions. 

    #### models package

    - The **model.py** file contains the **Model** parent class and associated functions for creating and evaluating survival models.
    - Specific survival model wrapper classes are in **accelerated_failure_time.py**, **cox_elastic_net.py**, and **random_survival_forest.py**.

    #### metrics package

    - The **ibs_differences_test.py** file contains the **IBSDifferenceTest** class, which can be used to evaluate the statistical significance of differences between two bipartitions.

