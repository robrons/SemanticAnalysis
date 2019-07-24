# Semantic Analysis Environment Setup
### Step 1 – Install the Python 3.7.* [.exe files](https://www.python.org/ftp/python/3.7.4/python-3.7.4-amd64.exe).
### Step 2 – Add the Python 3.7 Directory to your System Path Environment Variable
* Go to Control Panel –> System Properties –> Environment Variables and select the PATH variable from the list below:   
![Image not found](https://www.aaronstannard.com/images/image_44.png)
* Click Edit:  
![Image not found](https://www.aaronstannard.com/images/image_thumb_45.png)
* And append the Python path to the end of the string – the default path will be something like C:\Python37.
* Also make sure you include the C:\Python37\Scripts in the Path too even if it doesn’t exist yet – this is where your package management tools, unit testing tools, and other command line-accessible Python programs will live.
* With that in place, you can now start the Python interpreter on any command prompt by invoking the python command. Let’s get our package manager set up for Python.
### Step 3 – Install pip to Manage Your Python Packages
* [Pip has a detailed set of instructions on how to install it from source](http://www.pip-installer.org/en/latest/installing.html) – if you don’t have the curl command on your system, just use your Git or even your web browser to download the source file mentioned in their instructions.
### Step 4 - Installing required packages
* And now that you have pip up and running on your system, it’s trivial to install packages via the command line:
```
C:\> pip install numpy pandas sklearn csv
```
### Step 4 - Installing required packages
* And now that you have installed the required packages, you can run the program via the command line:
```
C:\> python Neural_Net_Single_Layer.py
```
