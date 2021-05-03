#make sure pip is in your path, for Windows Users

import sys
import subprocess
import pkg_resources

#list of required packages to be installed into system
required = {'numpy','pandas','GoogleNews','goose3','networkx','tqdm','matplotlib','seaborn','py3plex','louvain','python-igraph','leidenalg','torch','flair','nltk'} 

if required:
    # implement pip as a subprocess:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install',*required])
    
from flair.models import SequenceTagger
tagger = SequenceTagger.load('ner')