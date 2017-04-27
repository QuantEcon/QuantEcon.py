"""
Support functions to Support QuantEcon.notebooks

The purpose of these utilities is to implement simple support functions to allow for automatic downloading
of any support files (python modules, or data) that may be required to run demonstration notebooks.

Note
----
Files on the REMOTE Github Server can be organised into folders but they will end up at the root level of
when downloaded as a support File

"https://github.com/QuantEcon/QuantEcon.notebooks/raw/master/dependencies/mpi/something.py" --> ./something.py

TODO
----
1. Write Style guide for QuantEcon.notebook contributions
2. Write an interface for Dat Server
3. Platform Agnostic (replace wget usage)

"""

import os
import requests
import warnings

#-Remote Structure-#
REPO = "https://github.com/QuantEcon/QuantEcon.notebooks"
RAW = "raw"
BRANCH = "master"
FOLDER = "dependencies"          #Hard Coded Dependencies Folder on QuantEcon.notebooks


def fetch_nb_dependencies(files, repo=REPO, raw=RAW, branch=BRANCH, folder=FOLDER, overwrite=False, verbose=True):
    """
    Retrieve raw files from QuantEcon.notebooks or any other Github repo
    
    Parameters
    ----------
    file_list   list or dict
                A list of files to specify a collection of filenames	
                A dict of dir : list(files) to specify a directory
    repo        str, optional(default=REPO)
    raw 		str, optional(defualt=RAW)
    			This is here in case github changes access to their raw files through web links
    branch      str, optional(default=BRANCH)
    folder      str, optional(default=FOLDER)
    overwrite   bool, optional(default=False)
    verbose     bool, optional(default=True)

    """

    #-Generate Common Data Structure-#
    if type(files) == list:
        files = {"" : files}

    status = []

    #-Obtain each requested file-#
    for directory in files.keys():
        if directory != "":
            if verbose: print("Parsing directory: %s"%directory)
        for fl in files[directory]:
            if directory != "":
                fl = directory+"/"+fl
            #-Check for Local Copy of File (Default Behaviour is to Skip)-#
            if not overwrite:
                if os.path.isfile(fl):
                    if verbose: print("A file named %s already exists in the specified directory ... skipping download." % fl)
                    status.append(False)
                    continue
            else:
                if verbose: print("Overwriting file %s ..."%fl)
            if verbose: print("Fetching file: %s"%fl)
            #-Get file in OS agnostic way using requests-#
            url = "/".join([repo,raw,branch,folder,fl])
            r = requests.get(url)
            with open(fl, "wb") as fl:
                fl.write(r.content)
            status.append(True)

    return status
