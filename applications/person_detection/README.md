# Introduction

In this notebook, I present the research and development of a multi-label neural network model for object detection(not localization, just present or not present). Specifically, the targeted devices for deployment are microcontrollers, so the model must be constrained.

I intend to train a small model to detect whether a person, a car, or both are present in an image. Note that **the following scripts can be used for different categories and applications**. Please check the rest of the repository to learn more and see the deployment phase to different microcontrollers.



# Setup

First, let's import all the libraries we will need and set some configuration parameters.


```python
#Common imports
import numpy as np
import time, os, sys, random
import pathlib
import urllib.request
import shutil, zipfile, tarfile
import pickle
from collections import defaultdict

# Tables
import pandas as pd

#Deep learning
import tensorflow as tf
import tensorflow_addons as tfa # F1 metric
from tensorflow import keras
assert tf.__version__ >= "2.0"

#Reproducibility
random.seed(23)
np.random.seed(23)
tf.random.set_seed(23)

#Plotting
%matplotlib inline
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

#Dataset utilities
from pycocotools.coco import COCO
```

As you can see in the last import, we will use the `pycocotools` library to start building our data pipeline.

For reproducibility, here is some information of what I have installed:


```python
# Record package versions for reproducibility
print("OS: {}".format(os.name))
print("Python version: {}".format(sys.version))
print("Numpy: {}".format(np.__version__))
print("Tensorflow: {}".format(tf.__version__))
```

    OS: nt
    Python version: 3.8.5 | packaged by conda-forge | (default, Aug 29 2020, 00:43:28) [MSC v.1916 64 bit (AMD64)]
    Numpy: 1.19.1
    Tensorflow: 2.3.0
    

Finally, let's set up a couple of global variables


```python
ROOT_PATH = "."
DATA_DIR = os.path.join(ROOT_PATH, "data")
coco_year = 2017
```

# Get the data

### Considerations

Unfortunately, the ImageNet dataset does not contain a class "person", so instead we can use the [COCO dataset](https://cocodataset.org/#home). On top of having the person class, the dataset name means "Common Objects in COntext" so, in theory, the images will be a better representation of the reality the model will be exposed to in the final application.

The visual wake word dataset would have been useful for this application if it could be extended to more than one foreground class, but it served as inspiration.

### Why COCO?
The application proposed in the repository talks about detecting pedestrians and vehicles in crossroads, so using a more specialized dataset should be the right choice. On the other hand, this is meant to be a demo to test at home or show to your employer or stakeholders before proceeding with the final application. Therefore, the COCO dataset is a great choice due to its context-rich images. It should be as easy as feeding the model a different dataset during the transfer learning phase to apply it to a similar task. At most, tuning some hyperparameters for fine-tuning.

### Execution
The COCO dataset is designed for a variety of deep learning applications, but classification is not one of them, so we will need to play with the COCO API to get the data in the format we want for **multi-label** classification.

The following class provides an easy interface to do just that, from downloading the original annotations and images to the conversion to a `tf.dataset` or `torch.utils.data.DataLoader` object for an optimized data pipeline. The information and comments provided should be enough to understand its API but do not forget to check the rest of the repository for the standalone script and more usage information.


```python
class COCO_MLC():
    """
    COCO_C aims to convert the original COCO dataset into a classification problem. This approach
    makes the task easier and smaller models can be fit. Useful for constrained devices.
    
    Args:
        data_dir: Folder where all data will be downloaded
        year: COCO dataset year
    
    """
    def __init__(self, data_dir, year="2017"):
        self.data_dir = data_dir
        self.year = str(year)
        self.ANN_FOLDER = os.path.join(self.data_dir, "annotations")
        self.split_names = ["instances_train{}.json".format(self.year),
                            "instances_val{}.json".format(self.year)]
        self.coco_categories = []
        self.coco_objs = dict()
        self.datasets = []
        self.datasets_lens = []
        self.classes = []
        
        if not os.path.exists(self.data_dir):
            print("data_dir does not exist, creating directory now")
            os.makedirs(self.data_dir)
            
    def download_annotations(self, delete_zip=False):
        """
        Create annotations folder in data dir and download `year` COCO annotations.

        """
        ANN_ZIP_FILE_PATH = os.path.join(self.ANN_FOLDER, "annotations_train{}.zip".format(self.year))
        ANN_URL = r"http://images.cocodataset.org/annotations/annotations_trainval{}.zip".format(self.year)
        
        if not os.path.exists(self.ANN_FOLDER):
            print("Creating annotations folder: {}".format(self.ANN_FOLDER))
            os.makedirs(self.ANN_FOLDER)
        if not os.path.exists(ANN_ZIP_FILE_PATH):
            print("Downloading annotations...")
            with urllib.request.urlopen(ANN_URL) as resp, open(ANN_ZIP_FILE_PATH, 'wb') as out:
                shutil.copyfileobj(resp, out)
            print ("... done downloading.")
        
        print("Unzipping {}".format(ANN_ZIP_FILE_PATH))
        with zipfile.ZipFile(ANN_ZIP_FILE_PATH,"r") as zip_ref:
            for split in self.split_names:
                split_zip_path = os.path.join("annotations", split)
                split_zip_path = split_zip_path.replace("\\", "/") # Needed by zipfile
                zip_ref.extract(split_zip_path, self.data_dir)
        print ("... done unzipping")

        if delete_zip:
            print("Removing original zip file...")
            os.remove(ANN_ZIP_FILE_PATH)
        print("... done")
        
        # Let's create a list of categories for the user to check
        val_ann_file = os.path.join(self.ANN_FOLDER, self.split_names[1])
        coco_obj = COCO(val_ann_file)
        cats = coco_obj.loadCats(coco_obj.getCatIds())
        self.coco_categories = [cat['name'] for cat in cats]
        
        print("Download annotations done")
        
    def download_images(self, classes=[], threshold_areas=[], only_length=False, max_length=None,
                       add_negative_class=True, neg_classes=[]):
        """
        Download images from the desired classes and store them in different folders. For example, 
        after running `download_annotations` and then this function with "person" and "car" classes,
        we end up with a tree that looks as follows:

        -- data_dir
            |-- annotations
            |-- train
            |   |-- car
            |   `-- person
            `-- val
                |-- car
                `-- person

        
        Args:
            classes: classes from which to download images
            threshold_areas: mininum area percentage the desired foreground object
                            must have to be downloaded. A threshold must be provided for each
                            class
            only_length: when True, it does not download the images
            max_length: the max number of COCO annotations to scan for. By default scans all.
                        This is useful when you want to get only a few downloaded images when 
                        trying new 'threshold_area' values. Note that in COCO dataset there is 
                        usually more than one annotation per image, so this parameter is not 
                        the amount of images to be downloaded, although the more annotations 
                        you scan, the more images will be downloaded, allsatisfying the 
                        `threshold_area` constraint.
            add_negative_class: when True, it downloads images that do not correspond to any 
                        of the `classes`. It downloads as many as needed to have a balanced 
                        dataset.
            neg_classes: COCO categories that will end up in the negative class. If empty,
                        it will use all remaining categories not in `classes`. It will
                        contain equal amount of each negative class(balanced).
        
        Returns:
            A dictionary with keys "train" and "val" that contains per split per
            category data length
        
        """
        if not classes:
            return
        self.classes = classes
        split_dirs = ["train", "val"]
        data_lens = {"train":[], "val":[]}
        
        for split, split_dir in zip(self.split_names, split_dirs):
            split_path = os.path.join(self.ANN_FOLDER, split)
            coco = COCO(split_path) # Should we make these class attributes? What about memory?
            self.coco_objs[split_dir]=coco
            cat_ids = coco.getCatIds(self.classes)
            for cat_id, cat_name, threshold_area in zip(cat_ids, self.classes, threshold_areas):
                print("Downloading {} data for {} category".format(split_dir, cat_name))
                cat_path = os.path.join(self.data_dir, split_dir ,cat_name)
                os.makedirs(cat_path, exist_ok=True)
                
                # load annotations
                ann_ids = coco.getAnnIds(catIds=[cat_id])[:max_length]
                anns = coco.loadAnns(ann_ids)
                
                # Check area threshold and create img data list
                imgs_data=[]
                for ann in anns:
                    img_id = ann["image_id"]
                    img = coco.loadImgs([img_id])[0]
                    img_area = img["height"] * img["width"]
                    normalized_object_area = ann["area"]/img_area
                    if normalized_object_area > threshold_area:
                        if img not in imgs_data: # Same image can have several annotations
                            imgs_data.append(img)
                
                N = len(imgs_data)
                data_lens[split_dir].append(N)
                if only_length:
                    continue
                    
                # Download images
                tic = time.time()
                for i, img in enumerate(imgs_data):
                    fname = os.path.join(cat_path, img['file_name'])
                    if not os.path.exists(fname):
                        urllib.request.urlretrieve(img['coco_url'], fname)
                    print('Downloaded {}/{} images (t={:.2f}s)'.format(i+1, N, time.time()- tic), end="\r")
                print("\n")
                
            if add_negative_class:
                print("Downloading {} data for negative category".format(split_dir, cat_name))
                neg_path = os.path.join(self.data_dir, split_dir, "negative")
                os.makedirs(neg_path, exist_ok=True)
                
                # Obtain the number of images per class to get a balanced negative class
                if not neg_classes:
                    # all categories except the positive ones
                    all_cats = coco.loadCats(coco.getCatIds())
                    neg_classes = set([cat["name"] for cat in all_cats]) - set(self.classes)   
                    
                #neg_len = np.array(data_lens[split_dir], dtype=np.int64).mean(dtype=np.int64)
                neg_len = max(data_lens[split_dir])
                imgs_per_class = max(1, neg_len / len(neg_classes))
                
                # The COCO API has an attribute of type dictionary where each category_id key maps to
                # all images_ids of that class. Let's use that to make sets.
                
                # Get the positive image ids, we use sets to avoid duplicates.
                pos_classes_img_ids = set()
                for cat_id in cat_ids:
                    pos_classes_img_ids |= set(coco.catToImgs[cat_id])
                
                neg_classes_id = coco.getCatIds(neg_classes)
                
                # Find non negative images that do not contain positives and add `imgs_per_class` from
                # each subclass to the negative one.
                neg_images_ids = []
                for nclass_id in neg_classes_id: # TODO(Look for a better way)
                    n_subclass_imgs = set(coco.catToImgs[nclass_id])
                    n_subclass_imgs -= pos_classes_img_ids
                    imgs_to_sample = imgs_per_class if len(n_subclass_imgs)>imgs_per_class \
                                    else len(n_subclass_imgs)
                    n_subclass_imgs = random.sample(tuple(n_subclass_imgs), int(imgs_to_sample))
                    neg_images_ids.extend(n_subclass_imgs)
                
                neg_imgs_data = coco.loadImgs(neg_images_ids)
                N = len(neg_imgs_data)
                data_lens[split_dir].append(N)
                if only_length:
                    continue
                
                # Download negative images
                tic = time.time()
                for i, img in enumerate(neg_imgs_data):
                    fname = os.path.join(neg_path, img['file_name'])
                    if not os.path.exists(fname):
                        urllib.request.urlretrieve(img['coco_url'], fname)
                    print('Downloaded {}/{} images (t={:.2f}s)'.format(i+1, N, time.time()- tic), end="\r")
                print("\n")
        if add_negative_class:
            self.classes.append("negative")
        return data_lens
    
    
    def to_tf_dataset(self, img_size=(240, 320), channels=3, batch_size = 32, normalize=False, \
                      max_class_len=[]):
        """
        Get `tf.data.Dataset` object from all the images downloaded in a convenient format for
        the train and validation splits.
        
        Args:
            img_size: this is the size (heigh, width) the images will be resized to
            channels: indicates the desired number of color channels for the output images. 
                      Accepted values are: 0 for automatic, 1 for grayscale, 3 for RGB.
            batch_size: the number of examples per batch
            normalize: whether to normalize the images to the [-1,1] range
            max_class_len: If specified, limit the per class examples(chosen randomly) to a fixed number. 
                      Must be a list of two elements(train and validation max class lengths). If None, 
                      there is no limit.
            
        Returns:
            train_ds: tf.data.Dataset for the train split where each element is a tuple; the first
                      element is the images with shape(batch_size, img_height, img_width, channels)
                      and the second is the labels with shape (batch_size, num_classes). Each label
                      is a vector with ones on the category indexes the image corresponds to and
                      zero otherwise.
            val_ds: tf.data.Dataset for the validation split that shares the same format as `train_ds`
            class_names: array containing the class names, with its indexes matching the datasets
                      labels.
            datasets_lens: list containing the train and val dataset lengths(number of examples), in 
                      that order.
        """
        # If you know how to improve this function please let me know or make a pull request
        # For convenience, lets use pathlib
        data_dir = pathlib.Path(self.data_dir)
        #class_names = np.array(self.classes) # Alternative
        class_names = np.array(sorted([item.name for item in data_dir.glob('train/*')])) 
        
        #Let's define a couple of inner functions we will use later
        def to_img_name(s):
            # This is used to go from a full path like data/train/person/001.jpg
            # to just 001.jpg
            return str(s).split(os.path.sep)[-1]
        
        def process_path(file_path, img_width, img_height):
            # A function to load an image from a path
            img = tf.io.read_file(file_path)
            img = tf.image.decode_jpeg(img, channels=channels)
            img = tf.image.resize(img, [img_height, img_width])
            return tf.cast(img, dtype=tf.uint8)
        
        def configure_for_performance(ds, img_batch_size):
            # Configure a tf.data.Dataset for performance
            #ds = ds.cache() # This fills up too much memory, prefetch should be fine
            #ds = ds.shuffle(shuffle_buf_size) # already done before
            ds = ds.batch(img_batch_size)
            ds = ds.prefetch(buffer_size=AUTOTUNE)
            return ds
        
        AUTOTUNE = tf.data.experimental.AUTOTUNE
        split_dirs = ["train", "val"]
        datasets = [] # this is where the tf datasets will be stored
        self.datasets_lens = []
        for split_ix, split in enumerate(split_dirs):
            img_filen = [] # tuple list where each pair is (image_filename, class_index)
            for i, class_n in enumerate(class_names):
                # For each class, get a list of all images in the corresponding folder and
                # make a tuple with its class index.
                pattern = "{}/{}/*.jpg".format(split, class_n)
                class_filen = [(filen, i) for filen in map(to_img_name, data_dir.glob(pattern))]
                if max_class_len:
                    # To avoid bias, we shuffle the filenames before slicing
                    random.shuffle(class_filen)
                    class_filen = class_filen[:max_class_len[split_ix]]
                img_filen += class_filen
                # for example, an element would now look like ('000000003711.jpg', 0)
            
            # Now we have to merge duplicate image files with different labels. To do so, we can
            # use a dictionary where each image filename is a key and the values are a list of
            # class indexes
            merged_dict = defaultdict(list)
            for filen, label in img_filen:
                merged_dict[filen].append(label)
            #Lets convert it back to a list
            img_filen = list(merged_dict.items())
            # an element with more than one class would now look like ('000000005205.jpg', [0, 2])
            self.datasets_lens.append(len(img_filen))
            
            # Now we need to reconstruct the full image filepath(to one of them) and encode the
            # labels as "multi-label one hot"
            for ix, elem in enumerate(img_filen):
                filen, labels = elem
                # Images with multiple labels have multiple possible filepaths(i.e. 
                # they exist in different categories) so we will take the first one,
                # for example.        
                class_n = class_names[labels[0]]
                full_path = data_dir / split / class_n / filen
                
                # The label has 1s wherever the index class corresponds to the example
                label = np.zeros(len(class_names),)
                label[labels] = 1
                
                img_filen[ix] = [str(full_path), label]
            # an element with more than one class would now look like
            # ()'data\\train\\car\\000000003711.jpg', array([1., 0., 0.]))            
            # Let's start creating the filename dataset
            filenames, labels = zip(*img_filen)
            filenames, labels = list(filenames), np.array(labels, dtype=np.float32) # for tf.data
            dataset = tf.data.Dataset.from_tensor_slices( (filenames, labels) )
            # This is a good time to shuffle the dataset, before loading the images.
            # This allows us to shuffle with a buffer as big as len(filenames)
            dataset = dataset.shuffle(buffer_size = len(filenames))
            
            # Let's use process_path() to get the image dataset
            h, w = img_size
            dataset = dataset.map(lambda x, y: (process_path(x, w, h), y), \
                                  num_parallel_calls=AUTOTUNE)
            if normalize:
                normalize = lambda img : (tf.cast(img, tf.float32) / 127.5)-1
                dataset = dataset.map(lambda x, y: (normalize(x), y), \
                                  num_parallel_calls=AUTOTUNE)
            
            # Batch it and configure it for performance
            dataset = configure_for_performance(dataset, batch_size)
            
            datasets.append(dataset)
        
        self.datasets = datasets
        
        return (*self.datasets, class_names, self.datasets_lens)
    
    def to_torch_dataloader(self):
        assert True, "This function is not yet implemented."
        return

```

With that done, we can create a `COCO_MLC` instance and download the annotations of the original COCO dataset.


```python
coco_mlc = COCO_MLC(DATA_DIR, coco_year)
```


```python
coco_mlc.download_annotations(delete_zip=False)
```

    Unzipping .\data\annotations\annotations_train2017.zip
    ... done unzipping
    ... done
    loading annotations into memory...
    Done (t=0.70s)
    creating index...
    index created!
    Download annotations done
    

Now we can download images for the categories we want. In this case, "person" and "car" are the classes of interest. We can also specify the percentage of area `threshold_areas` the object of interest must have in the image to get rid of the images where it is too small for each class. By default, it will also download a negative class; this is useful because there could be no people or cars in the image, so it is helpful for the model to have a class to specify that situation.

Exploring the COCO dataset on the official website, we can see that there are considerably more person images(66k) than car images(12k), so to help reduce that imbalance we set a lower threshold area for the car class.


```python
classes = ["person", "car"]
threshold_areas = [0.05, 0.01]
class_lengths = coco_mlc.download_images(classes, threshold_areas=threshold_areas, \
                         add_negative_class=False) # negative class already downloaded
```

    loading annotations into memory...
    Done (t=21.84s)
    creating index...
    index created!
    Downloading train data for person category
    Downloaded 37585/37585 images (t=13.31s)
    
    Downloading train data for car category
    Downloaded 6263/6263 images (t=2.38s)
    
    loading annotations into memory...
    Done (t=0.71s)
    creating index...
    index created!
    Downloading val data for person category
    Downloaded 1565/1565 images (t=0.55s)
    
    Downloading val data for car category
    Downloaded 254/254 images (t=0.08s)
    
    

Let's check how many images per split and per class we have:


```python
class_lengths
```




    {'train': [37585, 6263], 'val': [1565, 254]}



Each list corresponds to the number of images of person, car, and negative classes on each split, respectively. It is clear we have an imbalanced dataset. There are several approaches to this problem, some of them are:

- Get more data, which if it were a commercial application should be seriously considered.
- Oversampling, making copies of the under-represented class images.
- Undersampling, deleting instances from the over-represented class.
- _Generate_ synthetic data, using algorithms like SMOTE or even a GAN.
- Setting class weights, although training may produce abrupt steps.

Iterating over this step, I have realized the car class is not entirely well represented in the dataset, as reflected in the precision and recall of that class, so we need to get more data first. If you browse the car folder, you will quickly see how some images have blurred or occluded cars; and that is not entirely bad to make the model more sensitive, but more samples with a car in the foreground will help improve performance and meet our end application goals. Although it doesn't have the context variety of the COCO dataset, the [Cars dataset](https://ai.stanford.edu/~jkrause/cars/car_dataset.html) should work just fine.

After that, since we will use transfer learning techniques we will stick to the undersampling option, limiting the number of examples for training to `6263+4000=10263` which is the minimum of examples for a class(car) plus the car images we will add from the Cars dataset. This means we will throw away a lot of precious data, but we can always come back and try an alternative if the application performance does not meet its goals, as we have just done. Nevertheless, with only three classes and a pre-trained model we should be all right now.

To get more data, we will download `4000` training samples and `500` validation samples. Let's write a function that will both download the images and copy them the respective folders `data_dir/train/car` and `data_dir/val/car`:


```python
def download_and_merge_car_dataset(data_dir, n_train, n_val, car_filenames_pkl=None):
    """
    Download the cars dataset (https://ai.stanford.edu/~jkrause/cars/car_dataset.html) and copy `n_train`
    examples to data_dir/train/car and `n_val` examples to data_dir/val/car. The data downloaded is only the
    train split from the cars dataset, which has 8144 examples. Therefore, `n_train + n_val` should not exceed
    that quantity to prevent overlapping.

    Args:
        data_dir: data folder path
        n_train: number of train examples to copy to the destination folder data_dir/train/car
        n_val: number of val examples to copy to the destination folder data_dir/val/car
        car_filenames_pkl: To ensure future calls don't copy different images to the destination folders, a
            list of filenames is stored in `data_dir` if None. If specified, the filenames list is loaded.
            The first call should create such list. Subsequent calls should load from disk. This is needed
            because the filenames are shuffled before copying.

    """
    dataset_path = os.path.join(data_dir, "cars_dataset")
    if not os.path.exists(dataset_path):
        print("Creating cars_dataset folder")
        os.makedirs(dataset_path)
    
    zip_filepath = os.path.join(dataset_path, "cars_train.tgz")
    zip_url = "http://imagenet.stanford.edu/internal/car196/cars_train.tgz"
    if not os.path.exists(zip_filepath):
        print("Downloading dataset...")
        with urllib.request.urlopen(zip_url) as resp, open(zip_filepath, 'wb') as out:
            shutil.copyfileobj(resp, out)
        print ("... done downloading.")

    print("Unzipping {}".format(zip_filepath))
    #with tarfile.open(zip_filepath, "r") as fc:
    #    fc.extractall(dataset_path)
    
    if car_filenames_pkl:
        print("Reading from existing filenames")
        with open(car_filenames_pkl, 'rb') as cf:
            car_filenames = pickle.load(cf)
    else:
        print("Creating car_filenames.pkl for reproducible future calls")
        dataset_path = pathlib.Path(dataset_path)
        print(dataset_path)
        car_filenames_gen = dataset_path.glob("cars_train/*")
        car_filenames = [str(fn).split(os.path.sep)[-1] for fn in car_filenames_gen]
        random.seed(23)
        random.shuffle(car_filenames)
        with open("car_filenames.pkl", "wb") as cf:
            pickle.dump(car_filenames, cf)

    cars_train = car_filenames[:n_train]
    cars_val = car_filenames[-n_val:]

    print("Copying {} train examples".format(n_train))
    for fn in cars_train:
        src = "{}/cars_dataset/cars_train/{}".format(data_dir, fn)
        dst = "{}/train/car/{}".format(data_dir, fn)
        shutil.copyfile(src, dst)
    
    print("Copying {} validation examples".format(n_val))
    for fn in cars_val:
        src = "{}/cars_dataset/cars_train/{}".format(data_dir, fn)
        dst = "{}/val/car/{}".format(data_dir, fn)
        shutil.copyfile(src, dst)

    print("Done")
```


```python
download_and_merge_car_dataset(DATA_DIR, n_train=4000, n_val=500, car_filenames_pkl=None)
```

    Unzipping .\data\cars_dataset\cars_train.tgz
    Creating car_filenames.pkl for reproducible future calls
    data\cars_dataset
    Copying 4000 train examples
    Copying 500 validation examples
    Done
    

With that done, since we are using tensorflow and keras, we create the `tf.data.Dataset`s for the train and validation splits. We want the images to be normalized in the range `[-1,1]` and since the main target for the application features a relatively powerful ARM Cortex-M7 processor we will leave the image channels to `3`(RGB). The image size was decided after model selection. Finally, as described just above, we can use the `max_class_len` parameter to limit the number of examples per class. For validation we will not apply that limit though(setting it to `None`), we will just have to choose wisely the metrics for model evaluation.

Note that this function scans for files the folder structure `COCO_MLC.download_images` created, and that's why it will also take into account the Cars dataset's images we have just added.


```python
max_examples = 10263
BATCH_SIZE = 32
IMG_SIZE = (224,224)
```


```python
dataset_params = dict(
    img_size = IMG_SIZE,
    channels = 3,
    normalize = True,
    batch_size = BATCH_SIZE,
    max_class_len = [max_examples, None]
)

train_ds, val_ds, class_names, datasets_lens = coco_mlc.to_tf_dataset(**dataset_params)
```

# Explore the data

Let's take a moment to actually see the data we are working with. The number of examples per class and split is


```python
def get_exs_per_class(dataset):
    class_counts = np.array([0, 0 , 0])
    for image_batch, label_batch in iter(dataset):
        for i in range(len(label_batch)):
            label = label_batch[i]
            label_ixs = tf.where(label).numpy()
            class_counts[label_ixs] += 1
    return class_counts
    
print("Train examples per class:  {}".format(get_exs_per_class(train_ds)))
print("Val examples per class:  {}".format(get_exs_per_class(val_ds)))
```

    Train examples per class:  [10263 10263 10263]
    Val examples per class:  [ 754 1018 1565]
    

The classes with the correct index are


```python
print(class_names)
```

    ['car' 'negative' 'person']
    

Great! Now let's see some images


```python
image_batch, label_batch = next(iter(val_ds))

plt.figure(figsize=(10, 13))
for i in range(12):
    ax = plt.subplot(4, 3, i + 1)
    img = ((image_batch[i].numpy()+1)*127.5).astype("uint8") # channels = 3
    plt.imshow(img) # channels = 3
    #img = (tf.squeeze(image_batch[i]).numpy()) # channels = 1
    #plt.imshow(img, cmap='gray') # channels = 1
    label = label_batch[i]
    label_ixs = tf.where(label).numpy()
    plt.title(str(class_names[label_ixs]).replace("[", "").replace("]",""))
    plt.axis("off")
plt.show()
```


    
![png](person-car-detection-research_files/person-car-detection-research_31_0.png)
    


As we can see, some images have multiple classes, and some just correspond to the negative one. Let's proceed to the model selection step.

# The model

While doing some research, I came across [these amazing tables](https://github.com/nerox8664/awesome-computer-vision-models) containing lists of computer vision models for different tasks, with some characteristics like performance or number of parameters. I exported the classification model table and filtered them with the following criteria:
- A microcontroller has limited flash and RAM memory, so a model with a low number of parameters is needed.
- They also have a limited clock speed, so the number of floating-point operations(FLOPS) should not be too high.
- Since we will be doing transfer learning, a pre-trained model should be available.

These are the top five models meeting those filters and sorted by top one error rate:


```python
best_small_models = pd.read_csv("parsed_classification_models.csv")[:5] # utils.ipynb has the filtering code
best_small_models.iloc[:5,1:]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Model</th>
      <th>Number of parameters</th>
      <th>FLOPS (millions)</th>
      <th>Top-1 Error</th>
      <th>Top-5 Error</th>
      <th>Year</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>EfficientNet-B0</td>
      <td>5,288,548</td>
      <td>414.31</td>
      <td>24.77</td>
      <td>7.52</td>
      <td>2019</td>
    </tr>
    <tr>
      <th>1</th>
      <td>NASNet-A 4@1056</td>
      <td>5,289,978</td>
      <td>584.90</td>
      <td>25.68</td>
      <td>8.16</td>
      <td>2017</td>
    </tr>
    <tr>
      <th>2</th>
      <td>MobileNet</td>
      <td>4,231,976</td>
      <td>579.80</td>
      <td>26.61</td>
      <td>8.95</td>
      <td>2017</td>
    </tr>
    <tr>
      <th>3</th>
      <td>MobileNetV2</td>
      <td>3,504,960</td>
      <td>329.36</td>
      <td>26.97</td>
      <td>8.87</td>
      <td>2018</td>
    </tr>
    <tr>
      <th>4</th>
      <td>ShuffleNetV2</td>
      <td>2,278,604</td>
      <td>149.72</td>
      <td>31.44</td>
      <td>11.63</td>
      <td>2018</td>
    </tr>
  </tbody>
</table>
</div>



The main target for this application is an STM32H7 microcontroller, with generous 2MB of flash memory and 1MB of RAM. Those are really high-end specifications for a microcontroller. Still, none of the models in that list would fit those constraints, at least not in their original form. Both [EfficentNet](https://arxiv.org/abs/1905.11946) and [NASNet](https://arxiv.org/abs/1707.07012) shine on mobile applications, but [MobileNet](https://arxiv.org/abs/1704.04861) and [MobileNetV2](https://arxiv.org/abs/1801.04381) apart from performing nearly as well, they propose two hyperparameters that affect the model complexity and capacity and, in turn, size and latency.

### MobileNets
The MobileNet architecture was first proposed by Google in 2017 and showed how small and quick models could achieve near state of the art performance. The main idea is the use of depthwise separable convolution layers instead of the standard ones: First, a filter is applied to **one and each** input channel(i.e. spatially). Then, a 1x1 filter is applied to all of the resulting feature maps across the channels(i.e. depthwise). For a more visual and rigorous explanation please refer to the paper or the repository.

It can be shown that the computational cost ratio between depthwise separable convolution and standard convolution is:

$$ r = \frac{1}{N} + \frac{1}{D^{2}_{k}} $$

Where $N$ is the number of output channels and $D_k$ is the kernel(filter) height and width. With $N$ being typically in the range of [32, 1024] and the kernel sizes for depthwise separable convolutions being always $3\times3$, they need 7 o 9 times fewer computations than standard convolutions. Of course, another way of reducing the computational cost would be feeding a smaller image to the input, which is also shown in the paper.

As for the model size, while MobileNet already has relatively few parameters(around four million) we can push them a lot further down with a "width multiplier": For a given layer and width multiplier $\alpha$, the number of input channels $M$ becomes $\alpha M$ and the number of output channels $N$ becomes $\alpha N$. It can also be shown that the computational cost and number of parameters is reduced by _roughly_ a factor of $\alpha^2$.

MobileNetV2 was proposed in 2019 and iterates over its predecessor adding new concepts like inverted residuals or linear bottlenecks, improving the performance on popular datasets and keeping, and __even smaller footprint__.

### Baseline model

The good thing is that Keras has pre-trained MobileNets(V1 and V2) with different `alpha`s, so we can use those pre-trained models and see which one fits our needs.

I've already gone ahead and trained a few baseline models with "out of the box" hyperparameters with different MobileNet versions and depths to get an idea of the performance we will be getting. I manually wrote the results in a CSV file which we will be filling with the different architectures and hyperparameters we try. This is what I got:


```python
model_selection = pd.read_csv("model_selection.csv")
model_selection
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Version</th>
      <th>alpha</th>
      <th>n_dense</th>
      <th>loss</th>
      <th>SizeFP16(MB)</th>
      <th>train_accuracy</th>
      <th>val_accuracy</th>
      <th>val_AUC</th>
      <th>val_macrof1</th>
      <th>val_precision</th>
      <th>val_recall</th>
      <th>OptTR</th>
      <th>OptFT</th>
      <th>lr_TR</th>
      <th>lr_FN</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>V1</td>
      <td>0.50</td>
      <td>256</td>
      <td>BCE</td>
      <td>1.871</td>
      <td>0.8407</td>
      <td>0.8171</td>
      <td>0.9422</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>Adam</td>
      <td>RMSProp</td>
      <td>0.0001</td>
      <td>0.00001</td>
    </tr>
    <tr>
      <th>1</th>
      <td>V2</td>
      <td>0.50</td>
      <td>256</td>
      <td>BCE</td>
      <td>2.031</td>
      <td>0.8594</td>
      <td>0.8298</td>
      <td>0.9524</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>Adam</td>
      <td>RMSProp</td>
      <td>0.0001</td>
      <td>0.00001</td>
    </tr>
    <tr>
      <th>2</th>
      <td>V2</td>
      <td>0.35</td>
      <td>256</td>
      <td>BCE</td>
      <td>1.465</td>
      <td>0.8436</td>
      <td>0.8290</td>
      <td>0.9494</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>Adam</td>
      <td>RMSProp</td>
      <td>0.0001</td>
      <td>0.00001</td>
    </tr>
  </tbody>
</table>
</div>



Loss `BCE` refers to binary cross-entropy. The optimizers and learning rates are different during transfer learning and fine-tuning. The `n_dense` refers to the number of neurons in the fully connected layer after the base model. The training accuracy is quite decent given that the COCO dataset is quite diverse and some of the images have blurred or occluded elements(try browsing the data folder if you've downloaded it). For a more context-specific scenario, like pedestrian and vehicle detection in crossroads, the model would definitely perform better if fed a good specific dataset. The validation accuracy(although slightly person-class biased) is quite close to the training one, showing it hasn't overfitted. We will be looking at different metrics for validation soon.

The model size though is clearly a problem: With `alpha` higher or equal than `0.5` the model is too close or surpasses the flash memory limit. Still, and this is what makes MobileNets really powerful, is that they scale down quite nicely; with `alpha=0.35` in MobileNetV2 and `n_dense=256` the model fits in less than 1.5MB, leaving 500 kilobytes for the application, which should be enough if it is not too complex. It is important to note that the size refers to the model _quantized_ to 16-bit floats, meaning that the original parameters have been shrunk from 32-bit floats to 16-bit ones. The precision loss is not really important during inference and it gives us a x2 size reduction, so it is definitely worth it. One can also quantize the model to 8-bit integers, which is what we will end up doing to get the smallest footprint, sacrificing as less performance as possible.

Although MobileNetV2 tend to work better the tensorflow lite for microcontrollers does not support all of the operations needed in that model(October 2020). Specifically, it is the number of convolutional depthwise channels that is limited to `256`. Thankfully, MobileNet with $\alpha=0.25$ will just meet that constraint.

### Loss and metrics

Before delving deeper into the model creation I would like to discuss the model's metrics and loss. For the metrics, we will be sticking with accuracy and F1 score. If you are not familiar with the latter, it is just the harmonic mean of the _precission_ and _recall_:

$$F_{1}=\frac{2}{\operatorname{recall}^{-1}+\text {precision }^{-1}}$$

If we decompose those terms in terms of true positives, false positives and false negatives, the F1 score looks like this:  

$$F_1 = \frac{2\mathrm{tp}}{2\mathrm{tp} + \mathrm{fp}+\mathrm{fn}}$$

The harmonic mean is the equivalent of the arithmetic mean for reciprocals of quantities that should be averaged by the arithmetic mean. To give some intuition, think about how the reciprocals of both precision and recall share the same denominator(true positives), so when we sum those quantities we are actually getting the arithmetic mean of false positives and false negatives over true positives. Finally, we take the reciprocal of that sum to transform it to the original representation. Visually, we see that to keep a high $F_1$ score we need an equilibrium between precision and recall:


```python
fig = plt.figure(figsize=(8,6))
ax = fig.gca(projection='3d')

precision = np.linspace(0.01, 1, 100)
recall = np.linspace(0.01,1,100)
pp, rr = np.meshgrid(precision, recall)
f1 = 2/((1/pp)+(1/rr))

surf =ax.plot_surface(pp, rr, f1, cmap=mpl.cm.coolwarm,
                linewidth=0, antialiased=False)

fig.colorbar(surf, shrink=0.5, aspect=5, label="$F_1$ score")
ax.set_xlabel("precision")
ax.set_ylabel("recall")
plt.show()
```


    
![png](person-car-detection-research_files/person-car-detection-research_42_0.png)
    


For the multi-label classification task we can just take the average of each class' $F_1$ score, which we will call "macro $F_1$ score". We will a threshold of `0.5` for this metric.

Going one step further, I found [this amazing post](https://towardsdatascience.com/the-unknown-benefits-of-using-a-soft-f1-loss-in-classification-systems-753902c0105d) explaining how one could even make a differentiable F1 function to use as loss. It also explains some of the benefits like optimizing directly for the macro $F_1$ score and the fact that the need for choosing a threshold to consider each class positive is implicitly inside the loss and we don't have to tweak those values. For more details and implementation details, I highly encourage you to check the post.

Nevertheless, although that sounds great we will also try the go-to loss function: binary cross-entropy, which maximizes the log-likelihood for each class and then we will see which one performs better for our task.

We can use the `metrics.F1Score` metric inside tensorflow-addons, but we'll have to define our own custom loss:


```python
class MacroDoubleSoftF1(keras.losses.Loss):
    """
    Compute the macro soft F1-score as a cost (average 1 - soft-F1 across all labels).
    Use probability values instead of binary predictions.
    This version uses the computation of soft-F1 for both positive and negative class for each label.
    Credits to ashrfem in github.
    
    Args:
        y (int32 Tensor): targets array of shape (BATCH_SIZE, N_LABELS)
        y_hat (float32 Tensor): probability matrix from forward propagation of shape (BATCH_SIZE, N_LABELS)
        
    Returns:
        cost (scalar Tensor): value of the cost function for the batch
    """
    def __init__(self, name="macro_double_soft_f1", **kwargs):
        super().__init__(name=name, **kwargs)
        
    def call(self, y_true, y_pred):
        y = tf.cast(y_true, tf.float32)
        y_hat = tf.cast(y_pred, tf.float32)
        
        tp = tf.reduce_sum(y_hat * y, axis=0)
        fp = tf.reduce_sum(y_hat * (1 - y), axis=0)
        fn = tf.reduce_sum((1 - y_hat) * y, axis=0)
        tn = tf.reduce_sum((1 - y_hat) * (1 - y), axis=0)
        
        soft_f1_class1 = 2*tp / (2*tp + fn + fp + 1e-16)
        soft_f1_class0 = 2*tn / (2*tn + fn + fp + 1e-16)
        
        cost_class1 = 1 - soft_f1_class1 # reduce 1 - soft-f1_class1 in order to increase soft-f1 on class 1
        cost_class0 = 1 - soft_f1_class0 # reduce 1 - soft-f1_class0 in order to increase soft-f1 on class 0
        cost = 0.5 * (cost_class1 + cost_class0) # take into account both class 1 and class 0
        macro_cost = tf.reduce_mean(cost) # average on all labels
        
        return macro_cost
```

### Implementation and hyperparameter tuning

Transfer learning is the process of using parts of a pretrained network and fine-tuning its parameters to a new task. A common practice with CNNs is to keep the convolutional layers(with their weights frozen) and add a classification head, typically consisting of a fully connected layer and then the output layer. It's good practice to add some kind of regularization to prevent overfitting, as it is especially common in transfer learning(big model, little data). After training the classification head, we can unfreeze the rest of the network, which we will call feature extractor, and fine tune those weights. This last process can push the performance up a little bit but can also quickly overfit the data, so it must be done with caution.

Let's build a function to create a model with different hyperparameters and perform transfer learning with fine-tuning.


```python
def build_model(pms, classification_report=True, class_names=[]):
    """
    Build and train a model with a given set of hyperparameters
    
    Args:
        pms: Dictionary with the hyperparameters, see examples for a template
        classification_report: wheter to return classification report generated by sklearn
        class_names: list with category names, used for the classification report
        
    Returns: 
        model: trained model
        history: data logged from keras model when fitting model during transfer learning
        history_fine: data logged from keras model when fitting model during fine tuning
    
    """
    # Load pretrained MobileNet on imageNet
    # MobileNets expect images with pixels in the range[-1,1]
    feature_extractor = keras.applications.MobileNet(input_shape=pms["img_shape"],
                                                alpha=pms["alpha"],
                                                weights='imagenet',
                                                include_top=False)
    # Freeze weights during transfer learning
    feature_extractor.trainable = False
    
    # Build the classification head
    pool_size = {(224,224,3):(7,7), (192,192,3):(6,6), (160,160,3):(5,5), (128,128,3):(4,4)}
    classification_head = keras.Sequential([
        keras.layers.InputLayer(feature_extractor.output.shape[1:]), # batch size doesn't have to be included
        #keras.layers.GlobalAveragePooling2D(), # instead of flatten 
        tf.keras.layers.AveragePooling2D(pool_size=pool_size[pms["img_shape"]]), #instead of flatten and returns 4D array, 
        # we use this because then we don't need to add reshape which adds SHAPE op, not supported by tflite micro
        #keras.layers.Dense(pms["n_dense"]), # no activation, like in the paper
        # tflite micro has issues with quantized fully connected so we make a FCN
        keras.layers.Conv2D(pms["n_dense"], kernel_size=1, padding='valid'),
        keras.layers.Dropout(pms["dropout"]),
        #keras.layers.Dense(pms["n_output"], activation="sigmoid") # multi label classification
        keras.layers.Conv2D(pms["n_output"], kernel_size=1, padding='valid', activation="sigmoid"),
        tf.keras.layers.Reshape([pms["n_output"]]) # we will get rid of this layer during deployment
    ], name="classification_head")
    
    # Build complete model using functional API
    inputs = tf.keras.Input(shape=pms["img_shape"])
    features = feature_extractor(inputs, training=False) # training=False to prevent BatchNorm parameters to be altered
    outputs = classification_head(features)
    model = tf.keras.Model(inputs, outputs)
    
    # Compile and fit
    tic = time.time()
    model.compile(optimizer=pms["base_opt"], loss=pms["loss"], metrics=pms["metrics"])
    print("Training classification head...\n")
    
    history = model.fit(pms["train_ds"], epochs=pms["base_epochs"], verbose=2,
                        validation_data=pms["val_ds"], callbacks=pms["callbacks"])
    print("\n\n")
    
    # Fine tune the model
    for layer in feature_extractor.layers[pms["fine_tune_layer_from"]:]:
        layer.trainable=True
    
    # Compile and fit
    fine_epochs = history.epoch[-1] + pms["fine_epochs"]
    model.compile(optimizer=pms["fine_opt"], loss=pms["loss"], metrics=pms["metrics"])
    print("Fine tuning...\n")
    
    history_fine = model.fit(pms["train_ds"], epochs=fine_epochs, validation_data=pms["val_ds"], verbose=2,
                             callbacks=pms["callbacks"], initial_epoch=history.epoch[-1])
    print("\n\n")
    
    delta = time.time() - tic
    print("Done, training took {}m {}s\n".format(int(delta//60), int(delta%60)))
    
    if classification_report:
        print("Classification report for validation split:")
        from sklearn.metrics import classification_report
        x_val, y_val, y_pred = [], [], []
        for x_batch, y_batch in pms["val_ds"]:
            x_val.extend(x_batch)
            y_val.extend(y_batch>0.5)
            y_pred.extend(model.predict(x_batch)>0.5)

        class_report = classification_report(y_val, y_pred, target_names=class_names, zero_division=1)
        print(class_report)
        
    return model, history, history_fine
```

#### Run 1

We'll use the custom loss we have defined and a dense layer with 256 neurons, with `dropout=0.3`. Let's see how it performs compared to our baselines.


```python
run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
run_logdir = os.path.join(".", "logs", run_id)
model_name = os.path.join("models", "model_{}.h5".format(run_id))
print(run_id)

CALLBACKS = [
    keras.callbacks.ModelCheckpoint(model_name, save_best_only=True),
    keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
    keras.callbacks.TensorBoard(run_logdir),
]

METRICS = [
    tf.keras.metrics.BinaryAccuracy(name="bin_accuracy"),
    tf.keras.metrics.Precision(name="precision"),
    tf.keras.metrics.Recall(name="recall"),
    tfa.metrics.F1Score(len(class_names), average="macro", name="MacroF1"),
]

pms_01 = dict(
    img_shape = IMG_SIZE + (3,),
    train_ds = train_ds,
    val_ds = val_ds,
    alpha = 0.25,
    n_dense = 256,
    n_output = 3,
    dropout = 0.3,
    base_opt = keras.optimizers.Adam(lr=1e-4),
    fine_opt = keras.optimizers.RMSprop(lr=1e-5),
    base_epochs = 25,
    fine_epochs= 25,
    loss = MacroDoubleSoftF1(),
    metrics = METRICS,
    callbacks = CALLBACKS,
    fine_tune_layer_from = 57 # feature_extractor has 87 layers
    
)
```

    run_2020_10_05-01_26_46
    


```python
model_01, history_01, history_fine_01= build_model(pms_01, class_names=class_names)
```

    Training classification head...
    
    Epoch 1/25
    WARNING:tensorflow:From C:\Users\Kique\Anaconda3\envs\tf23\lib\site-packages\tensorflow\python\ops\summary_ops_v2.py:1277: stop (from tensorflow.python.eager.profiler) is deprecated and will be removed after 2020-07-01.
    Instructions for updating:
    use `tf.profiler.experimental.stop` instead.
    

    WARNING:tensorflow:From C:\Users\Kique\Anaconda3\envs\tf23\lib\site-packages\tensorflow\python\ops\summary_ops_v2.py:1277: stop (from tensorflow.python.eager.profiler) is deprecated and will be removed after 2020-07-01.
    Instructions for updating:
    use `tf.profiler.experimental.stop` instead.
    

    WARNING:tensorflow:Callbacks method `on_train_batch_end` is slow compared to the batch time (batch time: 0.0160s vs `on_train_batch_end` time: 0.1000s). Check your callbacks.
    

    WARNING:tensorflow:Callbacks method `on_train_batch_end` is slow compared to the batch time (batch time: 0.0160s vs `on_train_batch_end` time: 0.1000s). Check your callbacks.
    

    949/949 - 31s - loss: 0.2220 - bin_accuracy: 0.8321 - precision: 0.7601 - recall: 0.7360 - MacroF1: 0.7595 - val_loss: 0.1539 - val_bin_accuracy: 0.8845 - val_precision: 0.8397 - val_recall: 0.8163 - val_MacroF1: 0.8262
    Epoch 2/25
    949/949 - 27s - loss: 0.1642 - bin_accuracy: 0.8683 - precision: 0.8080 - recall: 0.8009 - MacroF1: 0.8075 - val_loss: 0.1444 - val_bin_accuracy: 0.8864 - val_precision: 0.8483 - val_recall: 0.8112 - val_MacroF1: 0.8306
    Epoch 3/25
    949/949 - 27s - loss: 0.1552 - bin_accuracy: 0.8731 - precision: 0.8159 - recall: 0.8067 - MacroF1: 0.8152 - val_loss: 0.1357 - val_bin_accuracy: 0.8916 - val_precision: 0.8535 - val_recall: 0.8226 - val_MacroF1: 0.8396
    Epoch 4/25
    949/949 - 27s - loss: 0.1503 - bin_accuracy: 0.8761 - precision: 0.8200 - recall: 0.8120 - MacroF1: 0.8178 - val_loss: 0.1342 - val_bin_accuracy: 0.8922 - val_precision: 0.8442 - val_recall: 0.8379 - val_MacroF1: 0.8363
    Epoch 5/25
    949/949 - 25s - loss: 0.1478 - bin_accuracy: 0.8775 - precision: 0.8218 - recall: 0.8146 - MacroF1: 0.8198 - val_loss: 0.1352 - val_bin_accuracy: 0.8902 - val_precision: 0.8422 - val_recall: 0.8334 - val_MacroF1: 0.8363
    Epoch 6/25
    949/949 - 25s - loss: 0.1451 - bin_accuracy: 0.8789 - precision: 0.8228 - recall: 0.8182 - MacroF1: 0.8219 - val_loss: 0.1368 - val_bin_accuracy: 0.8870 - val_precision: 0.8368 - val_recall: 0.8298 - val_MacroF1: 0.8344
    Epoch 7/25
    949/949 - 28s - loss: 0.1442 - bin_accuracy: 0.8796 - precision: 0.8234 - recall: 0.8198 - MacroF1: 0.8215 - val_loss: 0.1308 - val_bin_accuracy: 0.8932 - val_precision: 0.8489 - val_recall: 0.8349 - val_MacroF1: 0.8397
    Epoch 8/25
    949/949 - 26s - loss: 0.1423 - bin_accuracy: 0.8806 - precision: 0.8251 - recall: 0.8210 - MacroF1: 0.8237 - val_loss: 0.1303 - val_bin_accuracy: 0.8935 - val_precision: 0.8514 - val_recall: 0.8325 - val_MacroF1: 0.8400
    Epoch 9/25
    949/949 - 28s - loss: 0.1414 - bin_accuracy: 0.8816 - precision: 0.8261 - recall: 0.8233 - MacroF1: 0.8256 - val_loss: 0.1303 - val_bin_accuracy: 0.8932 - val_precision: 0.8508 - val_recall: 0.8322 - val_MacroF1: 0.8374
    Epoch 10/25
    949/949 - 27s - loss: 0.1404 - bin_accuracy: 0.8823 - precision: 0.8263 - recall: 0.8257 - MacroF1: 0.8244 - val_loss: 0.1277 - val_bin_accuracy: 0.8940 - val_precision: 0.8473 - val_recall: 0.8397 - val_MacroF1: 0.8422
    Epoch 11/25
    949/949 - 27s - loss: 0.1392 - bin_accuracy: 0.8830 - precision: 0.8273 - recall: 0.8267 - MacroF1: 0.8259 - val_loss: 0.1314 - val_bin_accuracy: 0.8930 - val_precision: 0.8501 - val_recall: 0.8325 - val_MacroF1: 0.8396
    Epoch 12/25
    949/949 - 25s - loss: 0.1389 - bin_accuracy: 0.8826 - precision: 0.8266 - recall: 0.8261 - MacroF1: 0.8256 - val_loss: 0.1293 - val_bin_accuracy: 0.8954 - val_precision: 0.8572 - val_recall: 0.8310 - val_MacroF1: 0.8398
    Epoch 13/25
    949/949 - 25s - loss: 0.1380 - bin_accuracy: 0.8837 - precision: 0.8290 - recall: 0.8265 - MacroF1: 0.8271 - val_loss: 0.1362 - val_bin_accuracy: 0.8893 - val_precision: 0.8558 - val_recall: 0.8112 - val_MacroF1: 0.8316
    Epoch 14/25
    949/949 - 25s - loss: 0.1373 - bin_accuracy: 0.8847 - precision: 0.8312 - recall: 0.8271 - MacroF1: 0.8288 - val_loss: 0.1298 - val_bin_accuracy: 0.8936 - val_precision: 0.8551 - val_recall: 0.8277 - val_MacroF1: 0.8359
    Epoch 15/25
    949/949 - 28s - loss: 0.1374 - bin_accuracy: 0.8841 - precision: 0.8283 - recall: 0.8295 - MacroF1: 0.8273 - val_loss: 0.1302 - val_bin_accuracy: 0.8931 - val_precision: 0.8495 - val_recall: 0.8337 - val_MacroF1: 0.8345
    
    
    
    Fine tuning...
    
    Epoch 15/39
    WARNING:tensorflow:Callbacks method `on_train_batch_end` is slow compared to the batch time (batch time: 0.0220s vs `on_train_batch_end` time: 3.9220s). Check your callbacks.
    

    WARNING:tensorflow:Callbacks method `on_train_batch_end` is slow compared to the batch time (batch time: 0.0220s vs `on_train_batch_end` time: 3.9220s). Check your callbacks.
    

    949/949 - 31s - loss: 0.1384 - bin_accuracy: 0.8847 - precision: 0.8300 - recall: 0.8290 - MacroF1: 0.8280 - val_loss: 0.1303 - val_bin_accuracy: 0.8927 - val_precision: 0.8510 - val_recall: 0.8301 - val_MacroF1: 0.8394
    Epoch 16/39
    949/949 - 26s - loss: 0.1390 - bin_accuracy: 0.8841 - precision: 0.8303 - recall: 0.8263 - MacroF1: 0.8275 - val_loss: 0.1322 - val_bin_accuracy: 0.8919 - val_precision: 0.8519 - val_recall: 0.8259 - val_MacroF1: 0.8368
    Epoch 17/39
    949/949 - 28s - loss: 0.1384 - bin_accuracy: 0.8838 - precision: 0.8301 - recall: 0.8255 - MacroF1: 0.8268 - val_loss: 0.1288 - val_bin_accuracy: 0.8926 - val_precision: 0.8507 - val_recall: 0.8301 - val_MacroF1: 0.8377
    Epoch 18/39
    949/949 - 26s - loss: 0.1386 - bin_accuracy: 0.8840 - precision: 0.8294 - recall: 0.8271 - MacroF1: 0.8268 - val_loss: 0.1309 - val_bin_accuracy: 0.8926 - val_precision: 0.8529 - val_recall: 0.8271 - val_MacroF1: 0.8382
    Epoch 19/39
    949/949 - 27s - loss: 0.1380 - bin_accuracy: 0.8838 - precision: 0.8298 - recall: 0.8259 - MacroF1: 0.8267 - val_loss: 0.1291 - val_bin_accuracy: 0.8940 - val_precision: 0.8528 - val_recall: 0.8319 - val_MacroF1: 0.8374
    Epoch 20/39
    949/949 - 27s - loss: 0.1384 - bin_accuracy: 0.8840 - precision: 0.8308 - recall: 0.8253 - MacroF1: 0.8266 - val_loss: 0.1297 - val_bin_accuracy: 0.8931 - val_precision: 0.8547 - val_recall: 0.8265 - val_MacroF1: 0.8372
    Epoch 21/39
    949/949 - 26s - loss: 0.1375 - bin_accuracy: 0.8846 - precision: 0.8308 - recall: 0.8272 - MacroF1: 0.8278 - val_loss: 0.1306 - val_bin_accuracy: 0.8922 - val_precision: 0.8512 - val_recall: 0.8280 - val_MacroF1: 0.8372
    Epoch 22/39
    949/949 - 28s - loss: 0.1381 - bin_accuracy: 0.8844 - precision: 0.8295 - recall: 0.8286 - MacroF1: 0.8272 - val_loss: 0.1301 - val_bin_accuracy: 0.8925 - val_precision: 0.8529 - val_recall: 0.8268 - val_MacroF1: 0.8353
    
    
    
    Done, training took 11m 13s
    
    Classification report for validation split:
    WARNING:tensorflow:7 out of the last 9 calls to <function Model.make_predict_function.<locals>.predict_function at 0x0000024089F89160> triggered tf.function retracing. Tracing is expensive and the excessive number of tracings could be due to (1) creating @tf.function repeatedly in a loop, (2) passing tensors with different shapes, (3) passing Python objects instead of tensors. For (1), please define your @tf.function outside of the loop. For (2), @tf.function has experimental_relax_shapes=True option that relaxes argument shapes that can avoid unnecessary retracing. For (3), please refer to https://www.tensorflow.org/tutorials/customization/performance#python_or_tensor_args and https://www.tensorflow.org/api_docs/python/tf/function for  more details.
    

    WARNING:tensorflow:7 out of the last 9 calls to <function Model.make_predict_function.<locals>.predict_function at 0x0000024089F89160> triggered tf.function retracing. Tracing is expensive and the excessive number of tracings could be due to (1) creating @tf.function repeatedly in a loop, (2) passing tensors with different shapes, (3) passing Python objects instead of tensors. For (1), please define your @tf.function outside of the loop. For (2), @tf.function has experimental_relax_shapes=True option that relaxes argument shapes that can avoid unnecessary retracing. For (3), please refer to https://www.tensorflow.org/tutorials/customization/performance#python_or_tensor_args and https://www.tensorflow.org/api_docs/python/tf/function for  more details.
    

                  precision    recall  f1-score   support
    
             car       0.79      0.88      0.83       754
        negative       0.78      0.83      0.80      1018
          person       0.95      0.81      0.87      1565
    
       micro avg       0.85      0.83      0.84      3337
       macro avg       0.84      0.84      0.84      3337
    weighted avg       0.86      0.83      0.84      3337
     samples avg       0.87      0.84      0.82      3337
    
    

That is actually pretty good for a 1.5MB model and significantly better than our baseline models. The foreground class with lowest precision is `car`(and inherently highest recall), which means we may predict a car is in the image when that is not the case. In fact, if we were to develop the pedestrian and vehicle detection, I would have chosen an $F$ score that gives more importance to recall, since it is better that the model doesn't act on the street light(pedestrians and vehicles are frequent) too easily. If it was tuned for precision, we would miss more events and maybe act on that(we detect no pedestrians so we turn the street light to flashing amber for cars to proceed).

It seems like our model is already doing quite well as seen in all of the metrics, but I'm curious to try other hyperparameters.

Before continuing, let me define a small function to add the results to our `model_selection` table:


```python
model_selection.iloc[:,5:12] = model_selection.iloc[:,5:12].round(3)
# Quick and not elegant function to add rows to our model_selection table
def log_results(model, history, version="V1", alpha=0.35, size=1.465, opt=["Adam", "RMSProp"], lr=[1e-4,1e-5]):
    #n_dense = model.layers[-1].layers[-3].units
    n_dense = model.layers[-1].layers[-5].filters
    rlog = history.history
    log = [version, alpha, n_dense, model.loss.name, size, rlog["bin_accuracy"][0], rlog["val_bin_accuracy"][0],
           -1., rlog["val_MacroF1"][0], rlog["val_precision"][0], rlog["val_recall"][0], opt[0],opt[1],lr[0],lr[1]]
    return log

def add_entry(data=[]):
    model_selection.loc[len(model_selection)] = data
save_table = lambda: model_selection.to_csv("model_selection_complete.csv", index=False)
load_table = lambda: pd.read_csv("model_selection_complete.csv")
```


```python
add_entry(log_results(model_01, history_fine_01))
save_table()
```


```python
model_selection
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Version</th>
      <th>alpha</th>
      <th>n_dense</th>
      <th>loss</th>
      <th>SizeFP16(MB)</th>
      <th>train_accuracy</th>
      <th>val_accuracy</th>
      <th>val_AUC</th>
      <th>val_macrof1</th>
      <th>val_precision</th>
      <th>val_recall</th>
      <th>OptTR</th>
      <th>OptFT</th>
      <th>lr_TR</th>
      <th>lr_FN</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>V1</td>
      <td>0.50</td>
      <td>256</td>
      <td>BCE</td>
      <td>1.871</td>
      <td>0.841000</td>
      <td>0.817000</td>
      <td>0.942</td>
      <td>-1.000000</td>
      <td>-1.000000</td>
      <td>-1.000000</td>
      <td>Adam</td>
      <td>RMSProp</td>
      <td>0.0001</td>
      <td>0.00001</td>
    </tr>
    <tr>
      <th>1</th>
      <td>V2</td>
      <td>0.50</td>
      <td>256</td>
      <td>BCE</td>
      <td>2.031</td>
      <td>0.859000</td>
      <td>0.830000</td>
      <td>0.952</td>
      <td>-1.000000</td>
      <td>-1.000000</td>
      <td>-1.000000</td>
      <td>Adam</td>
      <td>RMSProp</td>
      <td>0.0001</td>
      <td>0.00001</td>
    </tr>
    <tr>
      <th>2</th>
      <td>V2</td>
      <td>0.35</td>
      <td>256</td>
      <td>BCE</td>
      <td>1.465</td>
      <td>0.844000</td>
      <td>0.829000</td>
      <td>0.949</td>
      <td>-1.000000</td>
      <td>-1.000000</td>
      <td>-1.000000</td>
      <td>Adam</td>
      <td>RMSProp</td>
      <td>0.0001</td>
      <td>0.00001</td>
    </tr>
    <tr>
      <th>3</th>
      <td>V1</td>
      <td>0.35</td>
      <td>256</td>
      <td>macro_double_soft_f1</td>
      <td>1.465</td>
      <td>0.884657</td>
      <td>0.892729</td>
      <td>-1.000</td>
      <td>0.839366</td>
      <td>0.850998</td>
      <td>0.830087</td>
      <td>Adam</td>
      <td>RMSProp</td>
      <td>0.0001</td>
      <td>0.00001</td>
    </tr>
  </tbody>
</table>
</div>



#### Run 2

Let's switch back to the binary cross-entropy to make sure the macro double F1 loss performs better. The rest of the hyperparameters are left intact.


```python
run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
run_logdir = os.path.join(".", "logs", run_id)
model_name = os.path.join("models", "model_{}.h5".format(run_id))
print(run_id)

CALLBACKS = [
    keras.callbacks.ModelCheckpoint(model_name, save_best_only=True),
    keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
    keras.callbacks.TensorBoard(run_logdir),
]

METRICS = [
    tf.keras.metrics.BinaryAccuracy(name="bin_accuracy"),
    tf.keras.metrics.Precision(name="precision"),
    tf.keras.metrics.Recall(name="recall"),
    tfa.metrics.F1Score(len(class_names), average="macro", name="MacroF1"),
]

pms_02 = dict(
    img_shape = IMG_SIZE + (3,),
    train_ds = train_ds,
    val_ds = val_ds,
    alpha = 0.25,
    n_dense = 256,
    n_output = 3,
    dropout = 0.3,
    base_opt = keras.optimizers.Adam(lr=1e-4),
    fine_opt = keras.optimizers.RMSprop(lr=1e-5),
    base_epochs = 25,
    fine_epochs= 25,
    loss = tf.keras.losses.BinaryCrossentropy(),
    metrics = METRICS,
    callbacks = CALLBACKS,
    fine_tune_layer_from = 57 # feature_extractor has 87 layers
    
)
```

    run_2020_10_05-01_44_12
    


```python
model_02, history_02, history_fine_02= build_model(pms_02, class_names=class_names)
```

    Training classification head...
    
    Epoch 1/25
    WARNING:tensorflow:Callbacks method `on_train_batch_end` is slow compared to the batch time (batch time: 0.0180s vs `on_train_batch_end` time: 6.1640s). Check your callbacks.
    

    WARNING:tensorflow:Callbacks method `on_train_batch_end` is slow compared to the batch time (batch time: 0.0180s vs `on_train_batch_end` time: 6.1640s). Check your callbacks.
    

    949/949 - 39s - loss: 0.3900 - bin_accuracy: 0.8282 - precision: 0.7676 - recall: 0.7060 - MacroF1: 0.7592 - val_loss: 0.2943 - val_bin_accuracy: 0.8739 - val_precision: 0.8409 - val_recall: 0.7761 - val_MacroF1: 0.8150
    Epoch 2/25
    949/949 - 32s - loss: 0.3190 - bin_accuracy: 0.8657 - precision: 0.8196 - recall: 0.7733 - MacroF1: 0.8085 - val_loss: 0.2628 - val_bin_accuracy: 0.8864 - val_precision: 0.8525 - val_recall: 0.8055 - val_MacroF1: 0.8326
    Epoch 3/25
    949/949 - 30s - loss: 0.3109 - bin_accuracy: 0.8696 - precision: 0.8249 - recall: 0.7800 - MacroF1: 0.8146 - val_loss: 0.2584 - val_bin_accuracy: 0.8910 - val_precision: 0.8663 - val_recall: 0.8037 - val_MacroF1: 0.8344
    Epoch 4/25
    949/949 - 33s - loss: 0.3055 - bin_accuracy: 0.8721 - precision: 0.8290 - recall: 0.7836 - MacroF1: 0.8177 - val_loss: 0.2577 - val_bin_accuracy: 0.8919 - val_precision: 0.8695 - val_recall: 0.8028 - val_MacroF1: 0.8363
    Epoch 5/25
    949/949 - 33s - loss: 0.3000 - bin_accuracy: 0.8732 - precision: 0.8305 - recall: 0.7854 - MacroF1: 0.8185 - val_loss: 0.2483 - val_bin_accuracy: 0.8966 - val_precision: 0.8698 - val_recall: 0.8187 - val_MacroF1: 0.8418
    Epoch 6/25
    949/949 - 30s - loss: 0.2982 - bin_accuracy: 0.8751 - precision: 0.8326 - recall: 0.7893 - MacroF1: 0.8202 - val_loss: 0.2528 - val_bin_accuracy: 0.8942 - val_precision: 0.8733 - val_recall: 0.8058 - val_MacroF1: 0.8433
    Epoch 7/25
    949/949 - 30s - loss: 0.2959 - bin_accuracy: 0.8766 - precision: 0.8353 - recall: 0.7911 - MacroF1: 0.8227 - val_loss: 0.2661 - val_bin_accuracy: 0.8866 - val_precision: 0.8599 - val_recall: 0.7965 - val_MacroF1: 0.8322
    Epoch 8/25
    949/949 - 28s - loss: 0.2956 - bin_accuracy: 0.8767 - precision: 0.8352 - recall: 0.7918 - MacroF1: 0.8230 - val_loss: 0.2541 - val_bin_accuracy: 0.8935 - val_precision: 0.8714 - val_recall: 0.8061 - val_MacroF1: 0.8412
    Epoch 9/25
    949/949 - 32s - loss: 0.2944 - bin_accuracy: 0.8769 - precision: 0.8356 - recall: 0.7919 - MacroF1: 0.8228 - val_loss: 0.2527 - val_bin_accuracy: 0.8939 - val_precision: 0.8644 - val_recall: 0.8160 - val_MacroF1: 0.8433
    Epoch 10/25
    949/949 - 31s - loss: 0.2930 - bin_accuracy: 0.8776 - precision: 0.8376 - recall: 0.7916 - MacroF1: 0.8227 - val_loss: 0.2696 - val_bin_accuracy: 0.8860 - val_precision: 0.8464 - val_recall: 0.8124 - val_MacroF1: 0.8307
    
    
    
    Fine tuning...
    
    Epoch 10/34
    WARNING:tensorflow:Callbacks method `on_train_batch_end` is slow compared to the batch time (batch time: 0.0140s vs `on_train_batch_end` time: 3.9190s). Check your callbacks.
    

    WARNING:tensorflow:Callbacks method `on_train_batch_end` is slow compared to the batch time (batch time: 0.0140s vs `on_train_batch_end` time: 3.9190s). Check your callbacks.
    

    949/949 - 35s - loss: 0.2969 - bin_accuracy: 0.8765 - precision: 0.8340 - recall: 0.7931 - MacroF1: 0.8226 - val_loss: 0.2633 - val_bin_accuracy: 0.8884 - val_precision: 0.8612 - val_recall: 0.8013 - val_MacroF1: 0.8321
    Epoch 11/34
    949/949 - 30s - loss: 0.2970 - bin_accuracy: 0.8754 - precision: 0.8338 - recall: 0.7890 - MacroF1: 0.8210 - val_loss: 0.2585 - val_bin_accuracy: 0.8910 - val_precision: 0.8639 - val_recall: 0.8067 - val_MacroF1: 0.8341
    Epoch 12/34
    949/949 - 32s - loss: 0.2957 - bin_accuracy: 0.8758 - precision: 0.8341 - recall: 0.7899 - MacroF1: 0.8228 - val_loss: 0.2601 - val_bin_accuracy: 0.8905 - val_precision: 0.8649 - val_recall: 0.8037 - val_MacroF1: 0.8333
    Epoch 13/34
    949/949 - 36s - loss: 0.2944 - bin_accuracy: 0.8768 - precision: 0.8360 - recall: 0.7909 - MacroF1: 0.8237 - val_loss: 0.2571 - val_bin_accuracy: 0.8916 - val_precision: 0.8661 - val_recall: 0.8061 - val_MacroF1: 0.8347
    Epoch 14/34
    949/949 - 32s - loss: 0.2960 - bin_accuracy: 0.8759 - precision: 0.8344 - recall: 0.7900 - MacroF1: 0.8206 - val_loss: 0.2598 - val_bin_accuracy: 0.8909 - val_precision: 0.8646 - val_recall: 0.8055 - val_MacroF1: 0.8342
    Epoch 15/34
    949/949 - 34s - loss: 0.2937 - bin_accuracy: 0.8771 - precision: 0.8354 - recall: 0.7930 - MacroF1: 0.8232 - val_loss: 0.2625 - val_bin_accuracy: 0.8883 - val_precision: 0.8644 - val_recall: 0.7968 - val_MacroF1: 0.8362
    Epoch 16/34
    949/949 - 34s - loss: 0.2938 - bin_accuracy: 0.8763 - precision: 0.8353 - recall: 0.7901 - MacroF1: 0.8240 - val_loss: 0.2563 - val_bin_accuracy: 0.8914 - val_precision: 0.8665 - val_recall: 0.8049 - val_MacroF1: 0.8362
    Epoch 17/34
    949/949 - 36s - loss: 0.2946 - bin_accuracy: 0.8763 - precision: 0.8339 - recall: 0.7918 - MacroF1: 0.8218 - val_loss: 0.2562 - val_bin_accuracy: 0.8925 - val_precision: 0.8667 - val_recall: 0.8085 - val_MacroF1: 0.8351
    Epoch 18/34
    949/949 - 36s - loss: 0.2929 - bin_accuracy: 0.8779 - precision: 0.8370 - recall: 0.7935 - MacroF1: 0.8223 - val_loss: 0.2571 - val_bin_accuracy: 0.8912 - val_precision: 0.8673 - val_recall: 0.8031 - val_MacroF1: 0.8356
    Epoch 19/34
    949/949 - 32s - loss: 0.2935 - bin_accuracy: 0.8770 - precision: 0.8359 - recall: 0.7919 - MacroF1: 0.8243 - val_loss: 0.2545 - val_bin_accuracy: 0.8932 - val_precision: 0.8698 - val_recall: 0.8070 - val_MacroF1: 0.8368
    Epoch 20/34
    949/949 - 34s - loss: 0.2938 - bin_accuracy: 0.8777 - precision: 0.8369 - recall: 0.7931 - MacroF1: 0.8234 - val_loss: 0.2609 - val_bin_accuracy: 0.8895 - val_precision: 0.8616 - val_recall: 0.8043 - val_MacroF1: 0.8355
    Epoch 21/34
    949/949 - 33s - loss: 0.2939 - bin_accuracy: 0.8777 - precision: 0.8369 - recall: 0.7929 - MacroF1: 0.8242 - val_loss: 0.2585 - val_bin_accuracy: 0.8916 - val_precision: 0.8642 - val_recall: 0.8085 - val_MacroF1: 0.8347
    Epoch 22/34
    949/949 - 34s - loss: 0.2928 - bin_accuracy: 0.8769 - precision: 0.8349 - recall: 0.7928 - MacroF1: 0.8243 - val_loss: 0.2584 - val_bin_accuracy: 0.8912 - val_precision: 0.8678 - val_recall: 0.8025 - val_MacroF1: 0.8363
    Epoch 23/34
    949/949 - 32s - loss: 0.2937 - bin_accuracy: 0.8773 - precision: 0.8361 - recall: 0.7928 - MacroF1: 0.8237 - val_loss: 0.2564 - val_bin_accuracy: 0.8925 - val_precision: 0.8655 - val_recall: 0.8100 - val_MacroF1: 0.8359
    Epoch 24/34
    949/949 - 32s - loss: 0.2922 - bin_accuracy: 0.8780 - precision: 0.8371 - recall: 0.7938 - MacroF1: 0.8238 - val_loss: 0.2568 - val_bin_accuracy: 0.8923 - val_precision: 0.8647 - val_recall: 0.8103 - val_MacroF1: 0.8359
    
    
    
    Done, training took 14m 27s
    
    Classification report for validation split:
                  precision    recall  f1-score   support
    
             car       0.81      0.87      0.84       754
        negative       0.82      0.80      0.81      1018
          person       0.95      0.78      0.86      1565
    
       micro avg       0.87      0.81      0.84      3337
       macro avg       0.86      0.82      0.83      3337
    weighted avg       0.88      0.81      0.84      3337
     samples avg       0.89      0.81      0.81      3337
    
    


```python
add_entry(log_results(model_02, history_fine_02))
save_table()
```


```python
model_selection
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Version</th>
      <th>alpha</th>
      <th>n_dense</th>
      <th>loss</th>
      <th>SizeFP16(MB)</th>
      <th>train_accuracy</th>
      <th>val_accuracy</th>
      <th>val_AUC</th>
      <th>val_macrof1</th>
      <th>val_precision</th>
      <th>val_recall</th>
      <th>OptTR</th>
      <th>OptFT</th>
      <th>lr_TR</th>
      <th>lr_FN</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>V1</td>
      <td>0.50</td>
      <td>256</td>
      <td>BCE</td>
      <td>1.871</td>
      <td>0.841000</td>
      <td>0.817000</td>
      <td>0.942</td>
      <td>-1.000000</td>
      <td>-1.000000</td>
      <td>-1.000000</td>
      <td>Adam</td>
      <td>RMSProp</td>
      <td>0.0001</td>
      <td>0.00001</td>
    </tr>
    <tr>
      <th>1</th>
      <td>V2</td>
      <td>0.50</td>
      <td>256</td>
      <td>BCE</td>
      <td>2.031</td>
      <td>0.859000</td>
      <td>0.830000</td>
      <td>0.952</td>
      <td>-1.000000</td>
      <td>-1.000000</td>
      <td>-1.000000</td>
      <td>Adam</td>
      <td>RMSProp</td>
      <td>0.0001</td>
      <td>0.00001</td>
    </tr>
    <tr>
      <th>2</th>
      <td>V2</td>
      <td>0.35</td>
      <td>256</td>
      <td>BCE</td>
      <td>1.465</td>
      <td>0.844000</td>
      <td>0.829000</td>
      <td>0.949</td>
      <td>-1.000000</td>
      <td>-1.000000</td>
      <td>-1.000000</td>
      <td>Adam</td>
      <td>RMSProp</td>
      <td>0.0001</td>
      <td>0.00001</td>
    </tr>
    <tr>
      <th>3</th>
      <td>V1</td>
      <td>0.35</td>
      <td>256</td>
      <td>macro_double_soft_f1</td>
      <td>1.465</td>
      <td>0.884657</td>
      <td>0.892729</td>
      <td>-1.000</td>
      <td>0.839366</td>
      <td>0.850998</td>
      <td>0.830087</td>
      <td>Adam</td>
      <td>RMSProp</td>
      <td>0.0001</td>
      <td>0.00001</td>
    </tr>
    <tr>
      <th>4</th>
      <td>V1</td>
      <td>0.35</td>
      <td>256</td>
      <td>binary_crossentropy</td>
      <td>1.465</td>
      <td>0.876545</td>
      <td>0.888447</td>
      <td>-1.000</td>
      <td>0.832091</td>
      <td>0.861192</td>
      <td>0.801319</td>
      <td>Adam</td>
      <td>RMSProp</td>
      <td>0.0001</td>
      <td>0.00001</td>
    </tr>
  </tbody>
</table>
</div>



This model seems to perform slightly worse in terms of macro $F1$ score and accuracy, and tuning for the adequate class thresholds would be more difficult. Therefore, it seems like the macro double soft $F_1$ loss is working quite well even though it is not widely known.

#### Run 3

With the macro double soft $F_1$ loss, let's try decreasing the number of units in the classification head


```python
run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
run_logdir = os.path.join(".", "logs", run_id)
model_name = os.path.join("models", "model_{}.h5".format(run_id))
print(run_id)

CALLBACKS = [
    keras.callbacks.ModelCheckpoint(model_name, save_best_only=True),
    keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
    keras.callbacks.TensorBoard(run_logdir),
]

METRICS = [
    tf.keras.metrics.BinaryAccuracy(name="bin_accuracy"),
    tf.keras.metrics.Precision(name="precision"),
    tf.keras.metrics.Recall(name="recall"),
    tfa.metrics.F1Score(len(class_names), average="macro", name="MacroF1"),
]

pms_03 = dict(
    img_shape = IMG_SIZE + (3,),
    train_ds = train_ds,
    val_ds = val_ds,
    alpha = 0.25,
    n_dense = 128,
    n_output = 3,
    dropout = 0.25,
    base_opt = keras.optimizers.Adam(lr=1e-4),
    fine_opt = keras.optimizers.RMSprop(lr=1e-5),
    base_epochs = 25,
    fine_epochs= 25,
    loss = MacroDoubleSoftF1(),
    metrics = METRICS,
    callbacks = CALLBACKS,
    fine_tune_layer_from = 57 # feature_extractor has 87 layers
    
)
```

    run_2020_10_05-02_04_02
    


```python
model_03, history_03, history_fine_03= build_model(pms_03, class_names=class_names)
```

    Training classification head...
    
    Epoch 1/25
    WARNING:tensorflow:Callbacks method `on_train_batch_end` is slow compared to the batch time (batch time: 0.0130s vs `on_train_batch_end` time: 5.7300s). Check your callbacks.
    

    WARNING:tensorflow:Callbacks method `on_train_batch_end` is slow compared to the batch time (batch time: 0.0130s vs `on_train_batch_end` time: 5.7300s). Check your callbacks.
    

    949/949 - 40s - loss: 0.2362 - bin_accuracy: 0.8217 - precision: 0.7453 - recall: 0.7182 - MacroF1: 0.7495 - val_loss: 0.1599 - val_bin_accuracy: 0.8791 - val_precision: 0.8333 - val_recall: 0.8058 - val_MacroF1: 0.8201
    Epoch 2/25
    949/949 - 31s - loss: 0.1703 - bin_accuracy: 0.8648 - precision: 0.8049 - recall: 0.7925 - MacroF1: 0.8042 - val_loss: 0.1473 - val_bin_accuracy: 0.8848 - val_precision: 0.8339 - val_recall: 0.8259 - val_MacroF1: 0.8313
    Epoch 3/25
    949/949 - 28s - loss: 0.1597 - bin_accuracy: 0.8709 - precision: 0.8118 - recall: 0.8047 - MacroF1: 0.8110 - val_loss: 0.1430 - val_bin_accuracy: 0.8888 - val_precision: 0.8501 - val_recall: 0.8172 - val_MacroF1: 0.8311
    Epoch 4/25
    949/949 - 32s - loss: 0.1547 - bin_accuracy: 0.8739 - precision: 0.8156 - recall: 0.8106 - MacroF1: 0.8151 - val_loss: 0.1436 - val_bin_accuracy: 0.8848 - val_precision: 0.8450 - val_recall: 0.8100 - val_MacroF1: 0.8250
    Epoch 5/25
    949/949 - 34s - loss: 0.1510 - bin_accuracy: 0.8753 - precision: 0.8175 - recall: 0.8128 - MacroF1: 0.8166 - val_loss: 0.1343 - val_bin_accuracy: 0.8923 - val_precision: 0.8506 - val_recall: 0.8292 - val_MacroF1: 0.8380
    Epoch 6/25
    949/949 - 34s - loss: 0.1480 - bin_accuracy: 0.8771 - precision: 0.8209 - recall: 0.8144 - MacroF1: 0.8198 - val_loss: 0.1370 - val_bin_accuracy: 0.8883 - val_precision: 0.8471 - val_recall: 0.8199 - val_MacroF1: 0.8310
    Epoch 7/25
    949/949 - 34s - loss: 0.1465 - bin_accuracy: 0.8780 - precision: 0.8210 - recall: 0.8174 - MacroF1: 0.8217 - val_loss: 0.1352 - val_bin_accuracy: 0.8893 - val_precision: 0.8454 - val_recall: 0.8256 - val_MacroF1: 0.8367
    Epoch 8/25
    949/949 - 31s - loss: 0.1447 - bin_accuracy: 0.8790 - precision: 0.8246 - recall: 0.8159 - MacroF1: 0.8214 - val_loss: 0.1369 - val_bin_accuracy: 0.8899 - val_precision: 0.8469 - val_recall: 0.8256 - val_MacroF1: 0.8333
    Epoch 9/25
    949/949 - 29s - loss: 0.1435 - bin_accuracy: 0.8798 - precision: 0.8239 - recall: 0.8200 - MacroF1: 0.8218 - val_loss: 0.1350 - val_bin_accuracy: 0.8898 - val_precision: 0.8469 - val_recall: 0.8253 - val_MacroF1: 0.8317
    Epoch 10/25
    949/949 - 29s - loss: 0.1423 - bin_accuracy: 0.8809 - precision: 0.8260 - recall: 0.8210 - MacroF1: 0.8237 - val_loss: 0.1383 - val_bin_accuracy: 0.8861 - val_precision: 0.8386 - val_recall: 0.8238 - val_MacroF1: 0.8295
    
    
    
    Fine tuning...
    
    Epoch 10/34
    WARNING:tensorflow:Callbacks method `on_train_batch_end` is slow compared to the batch time (batch time: 0.0150s vs `on_train_batch_end` time: 3.9270s). Check your callbacks.
    

    WARNING:tensorflow:Callbacks method `on_train_batch_end` is slow compared to the batch time (batch time: 0.0150s vs `on_train_batch_end` time: 3.9270s). Check your callbacks.
    

    949/949 - 36s - loss: 0.1487 - bin_accuracy: 0.8779 - precision: 0.8216 - recall: 0.8167 - MacroF1: 0.8198 - val_loss: 0.1374 - val_bin_accuracy: 0.8898 - val_precision: 0.8475 - val_recall: 0.8244 - val_MacroF1: 0.8336
    Epoch 11/34
    949/949 - 33s - loss: 0.1482 - bin_accuracy: 0.8773 - precision: 0.8205 - recall: 0.8156 - MacroF1: 0.8185 - val_loss: 0.1363 - val_bin_accuracy: 0.8899 - val_precision: 0.8482 - val_recall: 0.8238 - val_MacroF1: 0.8329
    Epoch 12/34
    949/949 - 33s - loss: 0.1479 - bin_accuracy: 0.8773 - precision: 0.8213 - recall: 0.8146 - MacroF1: 0.8190 - val_loss: 0.1370 - val_bin_accuracy: 0.8901 - val_precision: 0.8481 - val_recall: 0.8247 - val_MacroF1: 0.8337
    Epoch 13/34
    949/949 - 30s - loss: 0.1474 - bin_accuracy: 0.8777 - precision: 0.8216 - recall: 0.8156 - MacroF1: 0.8192 - val_loss: 0.1365 - val_bin_accuracy: 0.8897 - val_precision: 0.8490 - val_recall: 0.8220 - val_MacroF1: 0.8318
    Epoch 14/34
    949/949 - 30s - loss: 0.1480 - bin_accuracy: 0.8780 - precision: 0.8229 - recall: 0.8147 - MacroF1: 0.8197 - val_loss: 0.1379 - val_bin_accuracy: 0.8884 - val_precision: 0.8480 - val_recall: 0.8190 - val_MacroF1: 0.8317
    Epoch 15/34
    949/949 - 31s - loss: 0.1468 - bin_accuracy: 0.8788 - precision: 0.8233 - recall: 0.8172 - MacroF1: 0.8207 - val_loss: 0.1361 - val_bin_accuracy: 0.8899 - val_precision: 0.8478 - val_recall: 0.8244 - val_MacroF1: 0.8324
    Epoch 16/34
    949/949 - 30s - loss: 0.1465 - bin_accuracy: 0.8793 - precision: 0.8241 - recall: 0.8177 - MacroF1: 0.8199 - val_loss: 0.1351 - val_bin_accuracy: 0.8905 - val_precision: 0.8478 - val_recall: 0.8265 - val_MacroF1: 0.8339
    Epoch 17/34
    949/949 - 27s - loss: 0.1469 - bin_accuracy: 0.8780 - precision: 0.8217 - recall: 0.8163 - MacroF1: 0.8201 - val_loss: 0.1369 - val_bin_accuracy: 0.8894 - val_precision: 0.8467 - val_recall: 0.8241 - val_MacroF1: 0.8328
    Epoch 18/34
    949/949 - 27s - loss: 0.1462 - bin_accuracy: 0.8788 - precision: 0.8227 - recall: 0.8181 - MacroF1: 0.8213 - val_loss: 0.1362 - val_bin_accuracy: 0.8904 - val_precision: 0.8486 - val_recall: 0.8250 - val_MacroF1: 0.8333
    Epoch 19/34
    949/949 - 27s - loss: 0.1464 - bin_accuracy: 0.8789 - precision: 0.8240 - recall: 0.8164 - MacroF1: 0.8203 - val_loss: 0.1378 - val_bin_accuracy: 0.8892 - val_precision: 0.8468 - val_recall: 0.8232 - val_MacroF1: 0.8333
    Epoch 20/34
    949/949 - 29s - loss: 0.1463 - bin_accuracy: 0.8788 - precision: 0.8235 - recall: 0.8169 - MacroF1: 0.8209 - val_loss: 0.1372 - val_bin_accuracy: 0.8903 - val_precision: 0.8486 - val_recall: 0.8247 - val_MacroF1: 0.8335
    Epoch 21/34
    949/949 - 29s - loss: 0.1458 - bin_accuracy: 0.8792 - precision: 0.8250 - recall: 0.8161 - MacroF1: 0.8204 - val_loss: 0.1353 - val_bin_accuracy: 0.8895 - val_precision: 0.8480 - val_recall: 0.8226 - val_MacroF1: 0.8337
    
    
    
    Done, training took 12m 18s
    
    Classification report for validation split:
                  precision    recall  f1-score   support
    
             car       0.79      0.87      0.83       754
        negative       0.77      0.83      0.80      1018
          person       0.95      0.80      0.87      1565
    
       micro avg       0.85      0.83      0.84      3337
       macro avg       0.84      0.83      0.83      3337
    weighted avg       0.86      0.83      0.84      3337
     samples avg       0.87      0.83      0.82      3337
    
    


```python
add_entry(log_results(model_03, history_fine_03, size=1.144))
save_table()
```


```python
model_selection
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Version</th>
      <th>alpha</th>
      <th>n_dense</th>
      <th>loss</th>
      <th>SizeFP16(MB)</th>
      <th>train_accuracy</th>
      <th>val_accuracy</th>
      <th>val_AUC</th>
      <th>val_macrof1</th>
      <th>val_precision</th>
      <th>val_recall</th>
      <th>OptTR</th>
      <th>OptFT</th>
      <th>lr_TR</th>
      <th>lr_FN</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>V1</td>
      <td>0.50</td>
      <td>256</td>
      <td>BCE</td>
      <td>1.871</td>
      <td>0.841000</td>
      <td>0.817000</td>
      <td>0.942</td>
      <td>-1.000000</td>
      <td>-1.000000</td>
      <td>-1.000000</td>
      <td>Adam</td>
      <td>RMSProp</td>
      <td>0.0001</td>
      <td>0.00001</td>
    </tr>
    <tr>
      <th>1</th>
      <td>V2</td>
      <td>0.50</td>
      <td>256</td>
      <td>BCE</td>
      <td>2.031</td>
      <td>0.859000</td>
      <td>0.830000</td>
      <td>0.952</td>
      <td>-1.000000</td>
      <td>-1.000000</td>
      <td>-1.000000</td>
      <td>Adam</td>
      <td>RMSProp</td>
      <td>0.0001</td>
      <td>0.00001</td>
    </tr>
    <tr>
      <th>2</th>
      <td>V2</td>
      <td>0.35</td>
      <td>256</td>
      <td>BCE</td>
      <td>1.465</td>
      <td>0.844000</td>
      <td>0.829000</td>
      <td>0.949</td>
      <td>-1.000000</td>
      <td>-1.000000</td>
      <td>-1.000000</td>
      <td>Adam</td>
      <td>RMSProp</td>
      <td>0.0001</td>
      <td>0.00001</td>
    </tr>
    <tr>
      <th>3</th>
      <td>V1</td>
      <td>0.35</td>
      <td>256</td>
      <td>macro_double_soft_f1</td>
      <td>1.465</td>
      <td>0.884657</td>
      <td>0.892729</td>
      <td>-1.000</td>
      <td>0.839366</td>
      <td>0.850998</td>
      <td>0.830087</td>
      <td>Adam</td>
      <td>RMSProp</td>
      <td>0.0001</td>
      <td>0.00001</td>
    </tr>
    <tr>
      <th>4</th>
      <td>V1</td>
      <td>0.35</td>
      <td>256</td>
      <td>binary_crossentropy</td>
      <td>1.465</td>
      <td>0.876545</td>
      <td>0.888447</td>
      <td>-1.000</td>
      <td>0.832091</td>
      <td>0.861192</td>
      <td>0.801319</td>
      <td>Adam</td>
      <td>RMSProp</td>
      <td>0.0001</td>
      <td>0.00001</td>
    </tr>
    <tr>
      <th>5</th>
      <td>V1</td>
      <td>0.35</td>
      <td>128</td>
      <td>macro_double_soft_f1</td>
      <td>1.144</td>
      <td>0.877933</td>
      <td>0.889773</td>
      <td>-1.000</td>
      <td>0.833625</td>
      <td>0.847505</td>
      <td>0.824393</td>
      <td>Adam</td>
      <td>RMSProp</td>
      <td>0.0001</td>
      <td>0.00001</td>
    </tr>
  </tbody>
</table>
</div>



Halving the number of neurons has reduced the size and has similar macro $F_1$ score, not to mention the number of FLOPS have also been reduced.

#### Run 4

Let's decrease again `n_dense` to `64`


```python
run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
run_logdir = os.path.join(".", "logs", run_id)
model_name = os.path.join("models", "model_{}.h5".format(run_id))
print(run_id)

CALLBACKS = [
    keras.callbacks.ModelCheckpoint(model_name, save_best_only=True),
    keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
    keras.callbacks.TensorBoard(run_logdir),
]

METRICS = [
    tf.keras.metrics.BinaryAccuracy(name="bin_accuracy"),
    tf.keras.metrics.Precision(name="precision"),
    tf.keras.metrics.Recall(name="recall"),
    tfa.metrics.F1Score(len(class_names), average="macro", name="MacroF1"),
]

pms_04 = dict(
    img_shape = IMG_SIZE + (3,),
    train_ds = train_ds,
    val_ds = val_ds,
    alpha = 0.25,
    n_dense = 64,
    n_output = 3,
    dropout = 0.2,
    base_opt = keras.optimizers.Adam(lr=1e-4),
    fine_opt = keras.optimizers.RMSprop(lr=1e-5),
    base_epochs = 25,
    fine_epochs= 25,
    loss = MacroDoubleSoftF1(),
    metrics = METRICS,
    callbacks = CALLBACKS,
    fine_tune_layer_from = 57 # feature_extractor has 87 layers
    
)
```

    run_2020_10_05-03_05_32
    


```python
model_04, history_04, history_fine_04= build_model(pms_04, class_names=class_names)
```

    Training classification head...
    
    Epoch 1/25
    WARNING:tensorflow:Callbacks method `on_train_batch_end` is slow compared to the batch time (batch time: 0.0190s vs `on_train_batch_end` time: 14.4620s). Check your callbacks.
    

    WARNING:tensorflow:Callbacks method `on_train_batch_end` is slow compared to the batch time (batch time: 0.0190s vs `on_train_batch_end` time: 14.4620s). Check your callbacks.
    

    949/949 - 48s - loss: 0.2616 - bin_accuracy: 0.8030 - precision: 0.7228 - recall: 0.6774 - MacroF1: 0.7207 - val_loss: 0.1739 - val_bin_accuracy: 0.8737 - val_precision: 0.8246 - val_recall: 0.7986 - val_MacroF1: 0.8140
    Epoch 2/25
    949/949 - 29s - loss: 0.1792 - bin_accuracy: 0.8605 - precision: 0.7958 - recall: 0.7906 - MacroF1: 0.7984 - val_loss: 0.1530 - val_bin_accuracy: 0.8819 - val_precision: 0.8376 - val_recall: 0.8100 - val_MacroF1: 0.8231
    Epoch 3/25
    949/949 - 32s - loss: 0.1655 - bin_accuracy: 0.8672 - precision: 0.8053 - recall: 0.8011 - MacroF1: 0.8078 - val_loss: 0.1403 - val_bin_accuracy: 0.8912 - val_precision: 0.8450 - val_recall: 0.8331 - val_MacroF1: 0.8355
    Epoch 4/25
    949/949 - 30s - loss: 0.1588 - bin_accuracy: 0.8707 - precision: 0.8106 - recall: 0.8059 - MacroF1: 0.8108 - val_loss: 0.1446 - val_bin_accuracy: 0.8849 - val_precision: 0.8376 - val_recall: 0.8208 - val_MacroF1: 0.8302
    Epoch 5/25
    949/949 - 30s - loss: 0.1543 - bin_accuracy: 0.8735 - precision: 0.8134 - recall: 0.8124 - MacroF1: 0.8159 - val_loss: 0.1427 - val_bin_accuracy: 0.8841 - val_precision: 0.8364 - val_recall: 0.8196 - val_MacroF1: 0.8297
    Epoch 6/25
    949/949 - 33s - loss: 0.1515 - bin_accuracy: 0.8748 - precision: 0.8163 - recall: 0.8129 - MacroF1: 0.8167 - val_loss: 0.1380 - val_bin_accuracy: 0.8887 - val_precision: 0.8380 - val_recall: 0.8340 - val_MacroF1: 0.8320
    Epoch 7/25
    949/949 - 33s - loss: 0.1487 - bin_accuracy: 0.8774 - precision: 0.8205 - recall: 0.8162 - MacroF1: 0.8186 - val_loss: 0.1383 - val_bin_accuracy: 0.8884 - val_precision: 0.8452 - val_recall: 0.8229 - val_MacroF1: 0.8308
    Epoch 8/25
    949/949 - 33s - loss: 0.1476 - bin_accuracy: 0.8779 - precision: 0.8212 - recall: 0.8169 - MacroF1: 0.8198 - val_loss: 0.1387 - val_bin_accuracy: 0.8859 - val_precision: 0.8419 - val_recall: 0.8184 - val_MacroF1: 0.8318
    Epoch 9/25
    949/949 - 33s - loss: 0.1457 - bin_accuracy: 0.8786 - precision: 0.8224 - recall: 0.8178 - MacroF1: 0.8221 - val_loss: 0.1358 - val_bin_accuracy: 0.8903 - val_precision: 0.8530 - val_recall: 0.8187 - val_MacroF1: 0.8365
    Epoch 10/25
    949/949 - 30s - loss: 0.1447 - bin_accuracy: 0.8787 - precision: 0.8221 - recall: 0.8183 - MacroF1: 0.8221 - val_loss: 0.1327 - val_bin_accuracy: 0.8913 - val_precision: 0.8504 - val_recall: 0.8259 - val_MacroF1: 0.8381
    Epoch 11/25
    949/949 - 30s - loss: 0.1437 - bin_accuracy: 0.8802 - precision: 0.8242 - recall: 0.8210 - MacroF1: 0.8232 - val_loss: 0.1324 - val_bin_accuracy: 0.8937 - val_precision: 0.8491 - val_recall: 0.8364 - val_MacroF1: 0.8392
    Epoch 12/25
    949/949 - 29s - loss: 0.1434 - bin_accuracy: 0.8803 - precision: 0.8238 - recall: 0.8219 - MacroF1: 0.8239 - val_loss: 0.1364 - val_bin_accuracy: 0.8883 - val_precision: 0.8475 - val_recall: 0.8193 - val_MacroF1: 0.8336
    Epoch 13/25
    949/949 - 30s - loss: 0.1425 - bin_accuracy: 0.8803 - precision: 0.8232 - recall: 0.8228 - MacroF1: 0.8236 - val_loss: 0.1282 - val_bin_accuracy: 0.8958 - val_precision: 0.8596 - val_recall: 0.8292 - val_MacroF1: 0.8435
    Epoch 14/25
    949/949 - 30s - loss: 0.1417 - bin_accuracy: 0.8810 - precision: 0.8253 - recall: 0.8222 - MacroF1: 0.8247 - val_loss: 0.1335 - val_bin_accuracy: 0.8890 - val_precision: 0.8425 - val_recall: 0.8286 - val_MacroF1: 0.8331
    Epoch 15/25
    949/949 - 30s - loss: 0.1414 - bin_accuracy: 0.8813 - precision: 0.8254 - recall: 0.8233 - MacroF1: 0.8254 - val_loss: 0.1311 - val_bin_accuracy: 0.8957 - val_precision: 0.8567 - val_recall: 0.8328 - val_MacroF1: 0.8413
    Epoch 16/25
    949/949 - 30s - loss: 0.1404 - bin_accuracy: 0.8826 - precision: 0.8276 - recall: 0.8247 - MacroF1: 0.8252 - val_loss: 0.1337 - val_bin_accuracy: 0.8887 - val_precision: 0.8483 - val_recall: 0.8193 - val_MacroF1: 0.8334
    Epoch 17/25
    949/949 - 33s - loss: 0.1396 - bin_accuracy: 0.8827 - precision: 0.8275 - recall: 0.8254 - MacroF1: 0.8256 - val_loss: 0.1319 - val_bin_accuracy: 0.8926 - val_precision: 0.8542 - val_recall: 0.8253 - val_MacroF1: 0.8360
    Epoch 18/25
    949/949 - 30s - loss: 0.1399 - bin_accuracy: 0.8818 - precision: 0.8253 - recall: 0.8253 - MacroF1: 0.8265 - val_loss: 0.1337 - val_bin_accuracy: 0.8918 - val_precision: 0.8528 - val_recall: 0.8244 - val_MacroF1: 0.8339
    
    
    
    Fine tuning...
    
    Epoch 18/42
    WARNING:tensorflow:Callbacks method `on_train_batch_end` is slow compared to the batch time (batch time: 0.0180s vs `on_train_batch_end` time: 4.2450s). Check your callbacks.
    

    WARNING:tensorflow:Callbacks method `on_train_batch_end` is slow compared to the batch time (batch time: 0.0180s vs `on_train_batch_end` time: 4.2450s). Check your callbacks.
    

    949/949 - 38s - loss: 0.1417 - bin_accuracy: 0.8830 - precision: 0.8320 - recall: 0.8197 - MacroF1: 0.8256 - val_loss: 0.1306 - val_bin_accuracy: 0.8930 - val_precision: 0.8535 - val_recall: 0.8277 - val_MacroF1: 0.8386
    Epoch 19/42
    949/949 - 32s - loss: 0.1413 - bin_accuracy: 0.8820 - precision: 0.8267 - recall: 0.8239 - MacroF1: 0.8242 - val_loss: 0.1315 - val_bin_accuracy: 0.8917 - val_precision: 0.8510 - val_recall: 0.8265 - val_MacroF1: 0.8393
    Epoch 20/42
    949/949 - 31s - loss: 0.1408 - bin_accuracy: 0.8821 - precision: 0.8269 - recall: 0.8240 - MacroF1: 0.8267 - val_loss: 0.1321 - val_bin_accuracy: 0.8921 - val_precision: 0.8522 - val_recall: 0.8262 - val_MacroF1: 0.8395
    Epoch 21/42
    949/949 - 30s - loss: 0.1415 - bin_accuracy: 0.8813 - precision: 0.8257 - recall: 0.8228 - MacroF1: 0.8260 - val_loss: 0.1322 - val_bin_accuracy: 0.8921 - val_precision: 0.8525 - val_recall: 0.8259 - val_MacroF1: 0.8389
    Epoch 22/42
    949/949 - 28s - loss: 0.1407 - bin_accuracy: 0.8823 - precision: 0.8278 - recall: 0.8232 - MacroF1: 0.8248 - val_loss: 0.1318 - val_bin_accuracy: 0.8922 - val_precision: 0.8514 - val_recall: 0.8277 - val_MacroF1: 0.8393
    Epoch 23/42
    949/949 - 29s - loss: 0.1404 - bin_accuracy: 0.8824 - precision: 0.8266 - recall: 0.8254 - MacroF1: 0.8267 - val_loss: 0.1323 - val_bin_accuracy: 0.8917 - val_precision: 0.8505 - val_recall: 0.8271 - val_MacroF1: 0.8389
    
    
    
    Done, training took 13m 48s
    
    Classification report for validation split:
    WARNING:tensorflow:7 out of the last 11 calls to <function Model.make_predict_function.<locals>.predict_function at 0x00000240D8D90550> triggered tf.function retracing. Tracing is expensive and the excessive number of tracings could be due to (1) creating @tf.function repeatedly in a loop, (2) passing tensors with different shapes, (3) passing Python objects instead of tensors. For (1), please define your @tf.function outside of the loop. For (2), @tf.function has experimental_relax_shapes=True option that relaxes argument shapes that can avoid unnecessary retracing. For (3), please refer to https://www.tensorflow.org/tutorials/customization/performance#python_or_tensor_args and https://www.tensorflow.org/api_docs/python/tf/function for  more details.
    

    WARNING:tensorflow:7 out of the last 11 calls to <function Model.make_predict_function.<locals>.predict_function at 0x00000240D8D90550> triggered tf.function retracing. Tracing is expensive and the excessive number of tracings could be due to (1) creating @tf.function repeatedly in a loop, (2) passing tensors with different shapes, (3) passing Python objects instead of tensors. For (1), please define your @tf.function outside of the loop. For (2), @tf.function has experimental_relax_shapes=True option that relaxes argument shapes that can avoid unnecessary retracing. For (3), please refer to https://www.tensorflow.org/tutorials/customization/performance#python_or_tensor_args and https://www.tensorflow.org/api_docs/python/tf/function for  more details.
    

                  precision    recall  f1-score   support
    
             car       0.80      0.87      0.83       754
        negative       0.78      0.83      0.80      1018
          person       0.95      0.80      0.87      1565
    
       micro avg       0.85      0.83      0.84      3337
       macro avg       0.84      0.84      0.84      3337
    weighted avg       0.86      0.83      0.84      3337
     samples avg       0.88      0.83      0.82      3337
    
    


```python
add_entry(log_results(model_04, history_fine_04, size=0.984))
save_table()
```


```python
model_selection
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Version</th>
      <th>alpha</th>
      <th>n_dense</th>
      <th>loss</th>
      <th>SizeFP16(MB)</th>
      <th>train_accuracy</th>
      <th>val_accuracy</th>
      <th>val_AUC</th>
      <th>val_macrof1</th>
      <th>val_precision</th>
      <th>val_recall</th>
      <th>OptTR</th>
      <th>OptFT</th>
      <th>lr_TR</th>
      <th>lr_FN</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>V1</td>
      <td>0.50</td>
      <td>256</td>
      <td>BCE</td>
      <td>1.871</td>
      <td>0.841000</td>
      <td>0.817000</td>
      <td>0.942</td>
      <td>-1.000000</td>
      <td>-1.000000</td>
      <td>-1.000000</td>
      <td>Adam</td>
      <td>RMSProp</td>
      <td>0.0001</td>
      <td>0.00001</td>
    </tr>
    <tr>
      <th>1</th>
      <td>V2</td>
      <td>0.50</td>
      <td>256</td>
      <td>BCE</td>
      <td>2.031</td>
      <td>0.859000</td>
      <td>0.830000</td>
      <td>0.952</td>
      <td>-1.000000</td>
      <td>-1.000000</td>
      <td>-1.000000</td>
      <td>Adam</td>
      <td>RMSProp</td>
      <td>0.0001</td>
      <td>0.00001</td>
    </tr>
    <tr>
      <th>2</th>
      <td>V2</td>
      <td>0.35</td>
      <td>256</td>
      <td>BCE</td>
      <td>1.465</td>
      <td>0.844000</td>
      <td>0.829000</td>
      <td>0.949</td>
      <td>-1.000000</td>
      <td>-1.000000</td>
      <td>-1.000000</td>
      <td>Adam</td>
      <td>RMSProp</td>
      <td>0.0001</td>
      <td>0.00001</td>
    </tr>
    <tr>
      <th>3</th>
      <td>V1</td>
      <td>0.35</td>
      <td>256</td>
      <td>macro_double_soft_f1</td>
      <td>1.465</td>
      <td>0.884657</td>
      <td>0.892729</td>
      <td>-1.000</td>
      <td>0.839366</td>
      <td>0.850998</td>
      <td>0.830087</td>
      <td>Adam</td>
      <td>RMSProp</td>
      <td>0.0001</td>
      <td>0.00001</td>
    </tr>
    <tr>
      <th>4</th>
      <td>V1</td>
      <td>0.35</td>
      <td>256</td>
      <td>binary_crossentropy</td>
      <td>1.465</td>
      <td>0.876545</td>
      <td>0.888447</td>
      <td>-1.000</td>
      <td>0.832091</td>
      <td>0.861192</td>
      <td>0.801319</td>
      <td>Adam</td>
      <td>RMSProp</td>
      <td>0.0001</td>
      <td>0.00001</td>
    </tr>
    <tr>
      <th>5</th>
      <td>V1</td>
      <td>0.35</td>
      <td>128</td>
      <td>macro_double_soft_f1</td>
      <td>1.144</td>
      <td>0.877933</td>
      <td>0.889773</td>
      <td>-1.000</td>
      <td>0.833625</td>
      <td>0.847505</td>
      <td>0.824393</td>
      <td>Adam</td>
      <td>RMSProp</td>
      <td>0.0001</td>
      <td>0.00001</td>
    </tr>
    <tr>
      <th>6</th>
      <td>V1</td>
      <td>0.35</td>
      <td>64</td>
      <td>macro_double_soft_f1</td>
      <td>0.984</td>
      <td>0.884122</td>
      <td>0.894259</td>
      <td>-1.000</td>
      <td>0.838732</td>
      <td>0.853194</td>
      <td>0.832484</td>
      <td>Adam</td>
      <td>RMSProp</td>
      <td>0.0001</td>
      <td>0.00001</td>
    </tr>
  </tbody>
</table>
</div>



Again, halving the number of neurons in the classification head has considerably reduced size and FLOPS with performance is quite similar.

#### Run 5

So far the input image size has been `(224, 224, 3)` but we can try a lower resolution, like `(160, 160, 3)`. This greatly reduces inference time by a factor of $\rho = {\frac{160}{224}}^2 = 0.51$, meaning the model would be roughly two times faster.


```python
train_ds_small, val_ds_small, _, _ = coco_mlc.to_tf_dataset(img_size=(160,160), channels=3, normalize=True,
                                                            batch_size=BATCH_SIZE, max_class_len=[max_examples,None])
```


```python
run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
run_logdir = os.path.join(".", "logs", run_id)
model_name = os.path.join("models", "model_{}.h5".format(run_id))
print(run_id)

CALLBACKS = [
    keras.callbacks.ModelCheckpoint(model_name, save_best_only=True),
    keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
    keras.callbacks.TensorBoard(run_logdir),
]

METRICS = [
    tf.keras.metrics.BinaryAccuracy(name="bin_accuracy"),
    tf.keras.metrics.Precision(name="precision"),
    tf.keras.metrics.Recall(name="recall"),
    tfa.metrics.F1Score(len(class_names), average="macro", name="MacroF1"),
]

pms_05 = dict(
    img_shape = (160,160) + (3,),
    train_ds = train_ds_small,
    val_ds = val_ds_small,
    alpha = 0.25,
    n_dense = 64,
    n_output = 3,
    dropout = 0.2,
    base_opt = keras.optimizers.Adam(lr=1e-4),
    fine_opt = keras.optimizers.RMSprop(lr=1e-5),
    base_epochs = 25,
    fine_epochs= 25,
    loss = MacroDoubleSoftF1(),
    metrics = METRICS,
    callbacks = CALLBACKS,
    fine_tune_layer_from = 57 # feature_extractor has 87 layers
    
)
```

    run_2020_10_05-04_01_26
    


```python
model_05, history_05, history_fine_05= build_model(pms_05, class_names=class_names)
```

    Training classification head...
    
    Epoch 1/25
    WARNING:tensorflow:Callbacks method `on_train_batch_end` is slow compared to the batch time (batch time: 0.0140s vs `on_train_batch_end` time: 10.8920s). Check your callbacks.
    

    WARNING:tensorflow:Callbacks method `on_train_batch_end` is slow compared to the batch time (batch time: 0.0140s vs `on_train_batch_end` time: 10.8920s). Check your callbacks.
    

    949/949 - 39s - loss: 0.2543 - bin_accuracy: 0.8058 - precision: 0.7200 - recall: 0.6968 - MacroF1: 0.7287 - val_loss: 0.1750 - val_bin_accuracy: 0.8702 - val_precision: 0.8177 - val_recall: 0.7959 - val_MacroF1: 0.8046
    Epoch 2/25
    949/949 - 25s - loss: 0.1877 - bin_accuracy: 0.8516 - precision: 0.7843 - recall: 0.7741 - MacroF1: 0.7842 - val_loss: 0.1606 - val_bin_accuracy: 0.8747 - val_precision: 0.8158 - val_recall: 0.8160 - val_MacroF1: 0.8148
    Epoch 3/25
    949/949 - 26s - loss: 0.1746 - bin_accuracy: 0.8584 - precision: 0.7924 - recall: 0.7878 - MacroF1: 0.7926 - val_loss: 0.1543 - val_bin_accuracy: 0.8758 - val_precision: 0.8243 - val_recall: 0.8070 - val_MacroF1: 0.8198
    Epoch 4/25
    949/949 - 26s - loss: 0.1685 - bin_accuracy: 0.8614 - precision: 0.7972 - recall: 0.7915 - MacroF1: 0.7975 - val_loss: 0.1532 - val_bin_accuracy: 0.8793 - val_precision: 0.8299 - val_recall: 0.8115 - val_MacroF1: 0.8186
    Epoch 5/25
    949/949 - 27s - loss: 0.1642 - bin_accuracy: 0.8647 - precision: 0.7993 - recall: 0.8010 - MacroF1: 0.8022 - val_loss: 0.1458 - val_bin_accuracy: 0.8817 - val_precision: 0.8338 - val_recall: 0.8148 - val_MacroF1: 0.8244
    Epoch 6/25
    949/949 - 27s - loss: 0.1615 - bin_accuracy: 0.8653 - precision: 0.8007 - recall: 0.8011 - MacroF1: 0.8039 - val_loss: 0.1440 - val_bin_accuracy: 0.8824 - val_precision: 0.8319 - val_recall: 0.8202 - val_MacroF1: 0.8256
    Epoch 7/25
    949/949 - 28s - loss: 0.1593 - bin_accuracy: 0.8668 - precision: 0.8037 - recall: 0.8021 - MacroF1: 0.8042 - val_loss: 0.1464 - val_bin_accuracy: 0.8819 - val_precision: 0.8287 - val_recall: 0.8232 - val_MacroF1: 0.8200
    Epoch 8/25
    949/949 - 32s - loss: 0.1575 - bin_accuracy: 0.8677 - precision: 0.8043 - recall: 0.8048 - MacroF1: 0.8064 - val_loss: 0.1477 - val_bin_accuracy: 0.8776 - val_precision: 0.8279 - val_recall: 0.8085 - val_MacroF1: 0.8202
    Epoch 9/25
    949/949 - 27s - loss: 0.1561 - bin_accuracy: 0.8697 - precision: 0.8068 - recall: 0.8082 - MacroF1: 0.8076 - val_loss: 0.1444 - val_bin_accuracy: 0.8829 - val_precision: 0.8344 - val_recall: 0.8184 - val_MacroF1: 0.8241
    Epoch 10/25
    949/949 - 30s - loss: 0.1558 - bin_accuracy: 0.8694 - precision: 0.8074 - recall: 0.8063 - MacroF1: 0.8069 - val_loss: 0.1442 - val_bin_accuracy: 0.8812 - val_precision: 0.8277 - val_recall: 0.8220 - val_MacroF1: 0.8228
    Epoch 11/25
    949/949 - 29s - loss: 0.1542 - bin_accuracy: 0.8707 - precision: 0.8088 - recall: 0.8089 - MacroF1: 0.8104 - val_loss: 0.1415 - val_bin_accuracy: 0.8836 - val_precision: 0.8289 - val_recall: 0.8289 - val_MacroF1: 0.8269
    Epoch 12/25
    949/949 - 31s - loss: 0.1537 - bin_accuracy: 0.8707 - precision: 0.8082 - recall: 0.8100 - MacroF1: 0.8106 - val_loss: 0.1383 - val_bin_accuracy: 0.8862 - val_precision: 0.8362 - val_recall: 0.8277 - val_MacroF1: 0.8307
    Epoch 13/25
    949/949 - 28s - loss: 0.1525 - bin_accuracy: 0.8715 - precision: 0.8101 - recall: 0.8101 - MacroF1: 0.8108 - val_loss: 0.1387 - val_bin_accuracy: 0.8837 - val_precision: 0.8370 - val_recall: 0.8172 - val_MacroF1: 0.8266
    Epoch 14/25
    949/949 - 28s - loss: 0.1520 - bin_accuracy: 0.8719 - precision: 0.8107 - recall: 0.8106 - MacroF1: 0.8107 - val_loss: 0.1421 - val_bin_accuracy: 0.8823 - val_precision: 0.8301 - val_recall: 0.8226 - val_MacroF1: 0.8271
    Epoch 15/25
    949/949 - 29s - loss: 0.1514 - bin_accuracy: 0.8729 - precision: 0.8124 - recall: 0.8119 - MacroF1: 0.8114 - val_loss: 0.1423 - val_bin_accuracy: 0.8829 - val_precision: 0.8278 - val_recall: 0.8283 - val_MacroF1: 0.8274
    Epoch 16/25
    949/949 - 28s - loss: 0.1506 - bin_accuracy: 0.8728 - precision: 0.8119 - recall: 0.8120 - MacroF1: 0.8121 - val_loss: 0.1428 - val_bin_accuracy: 0.8825 - val_precision: 0.8300 - val_recall: 0.8235 - val_MacroF1: 0.8267
    Epoch 17/25
    949/949 - 28s - loss: 0.1493 - bin_accuracy: 0.8740 - precision: 0.8149 - recall: 0.8120 - MacroF1: 0.8133 - val_loss: 0.1406 - val_bin_accuracy: 0.8824 - val_precision: 0.8309 - val_recall: 0.8217 - val_MacroF1: 0.8258
    
    
    
    Fine tuning...
    
    Epoch 17/41
    WARNING:tensorflow:Callbacks method `on_train_batch_end` is slow compared to the batch time (batch time: 0.0140s vs `on_train_batch_end` time: 4.4830s). Check your callbacks.
    

    WARNING:tensorflow:Callbacks method `on_train_batch_end` is slow compared to the batch time (batch time: 0.0140s vs `on_train_batch_end` time: 4.4830s). Check your callbacks.
    

    949/949 - 32s - loss: 0.1523 - bin_accuracy: 0.8732 - precision: 0.8127 - recall: 0.8124 - MacroF1: 0.8125 - val_loss: 0.1424 - val_bin_accuracy: 0.8821 - val_precision: 0.8300 - val_recall: 0.8220 - val_MacroF1: 0.8238
    Epoch 18/41
    949/949 - 26s - loss: 0.1522 - bin_accuracy: 0.8719 - precision: 0.8103 - recall: 0.8114 - MacroF1: 0.8114 - val_loss: 0.1427 - val_bin_accuracy: 0.8830 - val_precision: 0.8338 - val_recall: 0.8196 - val_MacroF1: 0.8251
    Epoch 19/41
    949/949 - 30s - loss: 0.1515 - bin_accuracy: 0.8721 - precision: 0.8116 - recall: 0.8099 - MacroF1: 0.8107 - val_loss: 0.1429 - val_bin_accuracy: 0.8830 - val_precision: 0.8326 - val_recall: 0.8214 - val_MacroF1: 0.8249
    Epoch 20/41
    949/949 - 28s - loss: 0.1522 - bin_accuracy: 0.8719 - precision: 0.8109 - recall: 0.8102 - MacroF1: 0.8117 - val_loss: 0.1412 - val_bin_accuracy: 0.8836 - val_precision: 0.8329 - val_recall: 0.8229 - val_MacroF1: 0.8248
    Epoch 21/41
    949/949 - 31s - loss: 0.1520 - bin_accuracy: 0.8721 - precision: 0.8110 - recall: 0.8109 - MacroF1: 0.8113 - val_loss: 0.1422 - val_bin_accuracy: 0.8826 - val_precision: 0.8334 - val_recall: 0.8187 - val_MacroF1: 0.8244
    Epoch 22/41
    949/949 - 28s - loss: 0.1522 - bin_accuracy: 0.8722 - precision: 0.8122 - recall: 0.8094 - MacroF1: 0.8113 - val_loss: 0.1423 - val_bin_accuracy: 0.8838 - val_precision: 0.8338 - val_recall: 0.8223 - val_MacroF1: 0.8243
    Epoch 23/41
    949/949 - 28s - loss: 0.1519 - bin_accuracy: 0.8725 - precision: 0.8129 - recall: 0.8094 - MacroF1: 0.8112 - val_loss: 0.1439 - val_bin_accuracy: 0.8830 - val_precision: 0.8334 - val_recall: 0.8202 - val_MacroF1: 0.8242
    Epoch 24/41
    949/949 - 27s - loss: 0.1513 - bin_accuracy: 0.8726 - precision: 0.8124 - recall: 0.8104 - MacroF1: 0.8105 - val_loss: 0.1419 - val_bin_accuracy: 0.8838 - val_precision: 0.8342 - val_recall: 0.8217 - val_MacroF1: 0.8242
    Epoch 25/41
    949/949 - 27s - loss: 0.1517 - bin_accuracy: 0.8729 - precision: 0.8124 - recall: 0.8115 - MacroF1: 0.8108 - val_loss: 0.1407 - val_bin_accuracy: 0.8837 - val_precision: 0.8348 - val_recall: 0.8205 - val_MacroF1: 0.8251
    Epoch 26/41
    949/949 - 27s - loss: 0.1516 - bin_accuracy: 0.8729 - precision: 0.8125 - recall: 0.8113 - MacroF1: 0.8119 - val_loss: 0.1421 - val_bin_accuracy: 0.8846 - val_precision: 0.8350 - val_recall: 0.8235 - val_MacroF1: 0.8254
    Epoch 27/41
    949/949 - 30s - loss: 0.1515 - bin_accuracy: 0.8728 - precision: 0.8134 - recall: 0.8096 - MacroF1: 0.8112 - val_loss: 0.1428 - val_bin_accuracy: 0.8829 - val_precision: 0.8334 - val_recall: 0.8199 - val_MacroF1: 0.8246
    Epoch 28/41
    949/949 - 31s - loss: 0.1515 - bin_accuracy: 0.8729 - precision: 0.8126 - recall: 0.8112 - MacroF1: 0.8123 - val_loss: 0.1416 - val_bin_accuracy: 0.8837 - val_precision: 0.8325 - val_recall: 0.8238 - val_MacroF1: 0.8247
    Epoch 29/41
    949/949 - 26s - loss: 0.1510 - bin_accuracy: 0.8726 - precision: 0.8128 - recall: 0.8100 - MacroF1: 0.8114 - val_loss: 0.1415 - val_bin_accuracy: 0.8844 - val_precision: 0.8341 - val_recall: 0.8241 - val_MacroF1: 0.8266
    Epoch 30/41
    949/949 - 28s - loss: 0.1516 - bin_accuracy: 0.8724 - precision: 0.8121 - recall: 0.8101 - MacroF1: 0.8114 - val_loss: 0.1420 - val_bin_accuracy: 0.8839 - val_precision: 0.8338 - val_recall: 0.8226 - val_MacroF1: 0.8258
    
    
    
    Done, training took 15m 40s
    
    Classification report for validation split:
    WARNING:tensorflow:5 out of the last 107 calls to <function Model.make_predict_function.<locals>.predict_function at 0x00000240E4973C10> triggered tf.function retracing. Tracing is expensive and the excessive number of tracings could be due to (1) creating @tf.function repeatedly in a loop, (2) passing tensors with different shapes, (3) passing Python objects instead of tensors. For (1), please define your @tf.function outside of the loop. For (2), @tf.function has experimental_relax_shapes=True option that relaxes argument shapes that can avoid unnecessary retracing. For (3), please refer to https://www.tensorflow.org/tutorials/customization/performance#python_or_tensor_args and https://www.tensorflow.org/api_docs/python/tf/function for  more details.
    

    WARNING:tensorflow:5 out of the last 107 calls to <function Model.make_predict_function.<locals>.predict_function at 0x00000240E4973C10> triggered tf.function retracing. Tracing is expensive and the excessive number of tracings could be due to (1) creating @tf.function repeatedly in a loop, (2) passing tensors with different shapes, (3) passing Python objects instead of tensors. For (1), please define your @tf.function outside of the loop. For (2), @tf.function has experimental_relax_shapes=True option that relaxes argument shapes that can avoid unnecessary retracing. For (3), please refer to https://www.tensorflow.org/tutorials/customization/performance#python_or_tensor_args and https://www.tensorflow.org/api_docs/python/tf/function for  more details.
    

                  precision    recall  f1-score   support
    
             car       0.78      0.87      0.82       754
        negative       0.76      0.83      0.79      1018
          person       0.94      0.79      0.86      1565
    
       micro avg       0.83      0.82      0.83      3337
       macro avg       0.82      0.83      0.82      3337
    weighted avg       0.85      0.82      0.83      3337
     samples avg       0.86      0.83      0.81      3337
    
    

Performance has been sligthly decreased, but it is definitely worth the two times speed up. Therefore, this will be our final model.

### Where is the test set?

(TODO)

# Fine tuning for the application

(TODO get data from real distribution and retrain)

# Preparing for deployment

Before deployment, it is good practice to understand the necessary transformations to go from the raw camera data to the model input, but first let's load it and get a sample image for testing.


```python
#final_model = keras.models.load_model("./models/model_run_2020_10_05-03_05_32.h5", custom_objects={"MacroDoubleSoftF1":MacroDoubleSoftF1}) # 224,224
final_model = keras.models.load_model("./models/model_run_2020_10_05-04_01_26.h5", custom_objects={"MacroDoubleSoftF1":MacroDoubleSoftF1}) # 160,160
#final_model = keras.models.load_model("./models/model_run_2020_10_05-12_16_08.h5", custom_objects={"MacroDoubleSoftF1":MacroDoubleSoftF1}) # 128,128
```

Check performance is correct


```python
#metrics = final_model.evaluate(val_ds, verbose = 0)
metrics = final_model.evaluate(val_ds_small, verbose = 0)
```


```python
print("Loss: {:.2f}, Accuracy: {:.2f}, Precision: {:.2f}, Recall: {:.2f}, MacroF1: {:.2f}".format(*metrics))
```

    Loss: 0.14, Accuracy: 0.89, Precision: 0.84, Recall: 0.83, MacroF1: 0.83
    

Load an image with a person and a car for testing


```python
sample_img = mpl.image.imread(os.path.join(DATA_DIR,"val","car","000000383289.jpg")) # 000000383289

plt.figure()
plt.imshow(sample_img)
plt.axis("off")
plt.show()
```


    
![png](person-car-detection-research_files/person-car-detection-research_88_0.png)
    


And see its predictions


```python
IMG_SIZE = (160,160) # small model
```


```python
resized_img = tf.image.resize(sample_img, IMG_SIZE)/127.5 - 1
prediction = final_model.predict(resized_img[tf.newaxis, ...])
```


```python
print("Scores:\n Car: {:.2f}, Negative: {:.2f}, Person: {:.2f}".format(*prediction.flatten().tolist()))
```

    Scores:
     Car: 1.00, Negative: 0.00, Person: 1.00
    

Looks good, let's continue with our data pipeline to simulate microcontroller data.

### Dealing with the microcontroller data

> Note: We will be implementing part of the preprocessing in C since some of the ops and data types required are not supported by tensorflow lite. Nevertheless, let's do the whole process in python as well to explore and understand the data pipeline.

Microcontrollers have limited memory, so the cameras that interface with them usually have more compressed color formats than a normal computer would. Indeed, the camera I will be using, OV7670, has "YCbCr 4:2:2"(from which you could extract grayscale images) and RGB565 color modes. The standard RGB your computer uses for most of its work is 8 bits per color channel, so 24 bits are needed per pixel in RGB images. On the other hand, RGB565 uses 5 bits for red, 6 bits for green and another 5 bits for blue, for a total of 16 bits. This is quite convenient since one can store an image in `img_height*img_width /2` 32bit words.

Unfortunately, our model has an input shape of `(160,160,3)` for standard RGB images, so in a first instance we need to be able to convert between RGB888 and RGB555, so let's write a few functions so we can test it works:


```python
import tensorflow.bitwise as bw

def tf_rgb888_2_rgb565(img):
    # Expects a non normalized RGB image [0,255]
    img = tf.cast(img, tf.uint16)
    # Take most significant bits
    r = bw.right_shift(img[..., 0], 3) 
    g = bw.right_shift(img[..., 1], 2)
    b = bw.right_shift(img[..., 2], 3)

    # This is best seen as
    # img565 = (r<<11) | (g<<5) | (b<<0)
    # but in tensorflow ops looks like
    img565 = bw.bitwise_or( bw.bitwise_or(bw.left_shift(r, 11), 
            bw.left_shift(g, 5)), b)
    return img565
```


```python
def tf_rgb565_2_rgb888(img):
    # Mask each color channel and rescale to the range [0,255]
    r = bw.right_shift(bw.bitwise_and(img, 0xF800), 11) * 255//0x1F
    g = bw.right_shift(bw.bitwise_and(img, 0x7E0), 5) * 255//0x3F
    b = bw.bitwise_and(img, 0x1F) * 255//0x1F
    
    img888 = tf.stack([r, g, b], axis=-1)
    
    return img888
```

Let's try converting the sample image to RGB565, and back to RGB888 to check everything works as planned


```python
uc_img_size = (240, 320) # the resolution the OV7670 is configured for
sample_img_16 = tf.Variable(sample_img, dtype=tf.uint16)

img_res = tf.image.resize(sample_img_16, uc_img_size)
img_565 = tf.expand_dims(tf_rgb888_2_rgb565(img_res), axis=0)
img_565_flat = tf.reshape(img_565, (1, -1))
```


```python
img_888 = tf_rgb565_2_rgb888(img_565)
plt.imshow(tf.squeeze(img_888))
plt.axis("off")
plt.show()
```


    
![png](person-car-detection-research_files/person-car-detection-research_100_0.png)
    


Good, the image looks exactly the same, although in reality we have lost some information in the process: There are more possible colors in 24 bits than in 16 bits, but that shouldn't be a problem.  

As mentioned before we will do some of the preprocessing required in the C code, although it is helpful to see how our model performs on a more natural source of data(i.e raw RGB565 images), so we will do the whole process here as well.

With that said, we need to perform the following preprocessing steps:

1. Reshape a flat `uint16` input of shape `(img_height*img_width)` to `(img_height, img_width)`
2. Convert color from RGB565 to RGB888. Output shape is `(img_height, img_width, 3)`
3. Normalize the pixels to the range `[-1, 1]`
4. Resize the image to `(input_height, input_width)`. In this case, `(160,160)`. Output shape is equal to the model input shape, `(160,160,3)`

To do all those steps, let's create our own preprocessing layer:


```python
import tensorflow.bitwise as bw

class PreprocessRGB565(keras.layers.Layer):
    """
    Transforms an RGB565 image of shape `img_size` to an RGB888 normalized [-1, 1] image
    of size `out_img_size`
    
    """
    def __init__(self, img_size, out_img_size):
        super(PreprocessRGB565, self).__init__()
        self.height, self.width = img_size
        self.out_img_size = out_img_size
    
    def build(self, batch_input_shape):
        super().build(batch_input_shape)
    
    def call(self, inputs):
        # Reshape from (batch_size, height*width) to (batch_size, height, width)
        img_565 = tf.reshape(inputs, (-1, self.height, self.width))
        
        # We can convert the color and normalize to the range [-1,1] at the same time changing 
        # the scaling factor once the channel bits are masked. Channels red and blue range from
        # 0 to 2^5 -1 and channel green ranges from 0 to 2^6 -1.
        r = tf.cast(bw.right_shift(bw.bitwise_and(img_565, 0xF800), 11), tf.float32) /15.5 -1
        g = tf.cast(bw.right_shift(bw.bitwise_and(img_565, 0x7E0), 5), tf.float32) /31.5 -1
        b = tf.cast(bw.bitwise_and(img_565, 0x1F), tf.float32) /15.5 -1
        img_888 = tf.stack([r, g, b], axis=-1)
        
        # Resize the image
        res_img = tf.image.resize(img_888, self.out_img_size)
    
        return res_img
    
    def get_config(self):
        base_config = super().get_config()
        return {**base_config, "height": self.height, "width": self.width,
                "out_img_size": self.out_img_size}
    
    
```

We can now create our "microcontroller" model by adding an input layer and our preprocessing layer before the model:


```python
capture_height, capture_width = uc_img_size

uc_model = keras.models.Sequential([
    keras.layers.InputLayer((capture_height*capture_width), dtype=tf.uint16), # It is important to specify dtype
    PreprocessRGB565(img_size=uc_img_size, out_img_size=IMG_SIZE),
    final_model
])
```

If we now feed our RGB565 sample image to the network, we should get similar results:


```python
prediction = uc_model.predict(img_565_flat)
print("Scores:\n Car: {:.2f}, Negative: {:.2f}, Person: {:.2f}".format(*prediction.flatten().tolist()))
```

    Scores:
     Car: 1.00, Negative: 0.00, Person: 1.00
    

As explained above, the reconstructed image is not exactly the same to the original due to the smaller RGB format, but as we can see the model still predicts the same classes(remember the threshold is 0.5). If you are surprised the model is so confident on the class scores, it is due to the nature of the macro double soft F1 loss(again, I advise you to read [this post](https://towardsdatascience.com/the-unknown-benefits-of-using-a-soft-f1-loss-in-classification-systems-753902c0105d) and the paper linked there).

Let's double-check and evaluate this model on all validation examples


```python
def to_uc_format(img):
    # Transform each image to the microcontroller raw capture format
    non_norm_img = (img+1)*127.5
    resized_img = tf.image.resize(non_norm_img, uc_img_size)
    img_565 = tf_rgb888_2_rgb565(resized_img)
    img_565_flat = tf.reshape(img_565, (-1, 1))
    return img_565_flat
```


```python
uc_val_ds = val_ds.unbatch().map(lambda x, y: (to_uc_format(x), y)).batch(32)
```


```python
LOSS = MacroDoubleSoftF1()
METRICS = [
    tf.keras.metrics.BinaryAccuracy(name="bin_accuracy"),
    tf.keras.metrics.Precision(name="precision"),
    tf.keras.metrics.Recall(name="recall"),
    tfa.metrics.F1Score(len(class_names), average="macro", name="MacroF1"),
]

uc_model.compile(optimizer="SGD", loss=LOSS, metrics=METRICS) #arbitrary optimizer since we are not going to train it
```


```python
metrics = uc_model.evaluate(uc_val_ds, verbose=0)
print("Loss: {:.2f}, Accuracy: {:.2f}, Precision: {:.2f}, Recall: {:.2f}, MacroF1: {:.2f}".format(*metrics))
```

    Loss: 0.14, Accuracy: 0.88, Precision: 0.83, Recall: 0.82, MacroF1: 0.83
    

The performance has slightly decreased, but nothing to worry about; Seems to work quite well. We can confirm this data pipeline is correct and that the model performs well under those conditions.

### A small preprocess layer to include inside our model

Although some operations are not supported by tensorflow lite(or micro), we can get away with image resizing and normalization to make our C code less complex. Let's add that component to our final model by simplifying  the previous preprocess layer:


```python
import tensorflow.bitwise as bw

class uc_preprocess(keras.layers.Layer):
    """
    Input is a `(*img_size, 3)` uint8 array
    
    1. Transform uint8 input to float32
    2. Resize image from `img_size` to `out_img_size`
    3. Normalize to the range [-1,1]
    
    """
    def __init__(self, img_size, out_img_size, **kwargs):
        super(uc_preprocess, self).__init__(**kwargs)
        self.height, self.width = img_size
        self.out_img_size = out_img_size 
    
    def build(self, batch_input_shape):
        super().build(batch_input_shape)
    
    def call(self, inputs):
        # Cast from uint8 to float32 not supported by tflite for microcontrollers as of October, 2020
        # Workaround is to quantize the inputs during compression
        #imgs = tf.cast(inputs, tf.float32) 
        # Resize the image
        res_imgs = tf.image.resize(inputs, self.out_img_size, method='nearest') # method supported by tflite micro
        # Normalize to the range [-1,1]
        norm_imgs = res_imgs*(1/127.5) -1 # multiply reciprocal as DIV is not supported by tflite micro as of October, 2020
    
        return norm_imgs
    
    def get_config(self):
        base_config = super().get_config()
        return {**base_config, "height": self.height, "width": self.width,
                "out_img_size": self.out_img_size}
```


```python
#Another similar way to do it with a Lambda layer
def lm_uc_preprocess(inputs):
    res_imgs = tf.image.resize(inputs, IMG_SIZE, method='nearest') # method supported by tflite micro
    # Normalize to the range [-1,1]
    norm_imgs = res_imgs*(1/127.5) -1
    return norm_imgs
```


```python
capture_height, capture_width = uc_img_size

uc_final_model = keras.models.Sequential([
    keras.layers.InputLayer((capture_height, capture_width, 3), dtype=tf.float32),
    uc_preprocess(img_size=uc_img_size, out_img_size=IMG_SIZE), # (240, 320) to (224, 224)
    #keras.layers.Lambda(lm_uc_preprocess),
    *final_model.layers[:-1], # final reshape bugs tflite
    *final_model.layers[-1].layers[:-1]
])
```

Let's check it works


```python
test_image= tf.expand_dims(tf.image.resize(sample_img, uc_img_size), axis=0)
test_image.shape, test_image.dtype
```




    (TensorShape([1, 240, 320, 3]), tf.float32)




```python
prediction = uc_final_model.predict(test_image)
print("Scores:\n Car: {:.2f}, Negative: {:.2f}, Person: {:.2f}".format(*prediction.flatten().tolist()))
```

    Scores:
     Car: 1.00, Negative: 0.00, Person: 1.00
    

Nice, we can now convert this model.

### Compress the model


```python
def to_input_format(img):
    # Transform each image to the microcontroller raw capture format
    non_norm_img = (img+1)*127.5 # uint8 range 
    resized_img = tf.image.resize(non_norm_img, uc_img_size)
    return resized_img
```


```python
input_train_ds = train_ds.unbatch().map(lambda x, y: (to_input_format(x), y)).batch(1)
input_val_ds = val_ds.unbatch().map(lambda x, y: (to_input_format(x), y)).batch(1)
```

It is time to compress our model to use 8 bit integer parameters to reduce its size and save it in a `tflite` format. To quantize variable data, such as the input or activations, we can provide a representative dataset to estimate the dynamic range of the data.


```python
def representative_data_gen():
    for input_value, label_value in input_train_ds.take(30000):
        yield [input_value]
```


```python
def compress_and_save(model, name, representative_dataset_gen):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.inference_input_type= tf.int8 # has to be signed (tflite micro requirement)
    converter.inference_output_type = tf.int8
    converter.representative_dataset = representative_data_gen
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    
    with open(name, 'wb') as f:
        f.write(tflite_model)
        
    return tflite_model
```


```python
lite_model = compress_and_save(uc_final_model, "uc_final_model.tflite", representative_data_gen)
```

    WARNING:tensorflow:From C:\Users\Kique\Anaconda3\envs\tf23\lib\site-packages\tensorflow\python\training\tracking\tracking.py:111: Model.state_updates (from tensorflow.python.keras.engine.training) is deprecated and will be removed in a future version.
    Instructions for updating:
    This property should not be used in TensorFlow 2.0, as updates are applied automatically.
    WARNING:tensorflow:From C:\Users\Kique\Anaconda3\envs\tf23\lib\site-packages\tensorflow\python\training\tracking\tracking.py:111: Layer.updates (from tensorflow.python.keras.engine.base_layer) is deprecated and will be removed in a future version.
    Instructions for updating:
    This property should not be used in TensorFlow 2.0, as updates are applied automatically.
    INFO:tensorflow:Assets written to: C:\Users\Kique\AppData\Local\Temp\tmpco_f4jv4\assets
    

As always, let's check the model is still working as expected, but this time with the tflite API:


```python
# Load the TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="uc_final_model.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
```


```python
input_type = interpreter.get_input_details()[0]['dtype']
print('input type: ', input_type)
output_type = interpreter.get_output_details()[0]['dtype']
print('output type: ', output_type)
```

    input type:  <class 'numpy.int8'>
    output type:  <class 'numpy.int8'>
    


```python
# Test the model on random input data.
input_data = tf.cast(test_image-128, tf.int8)
interpreter.set_tensor(input_details[0]['index'], input_data)

interpreter.invoke()

# The function `get_tensor()` returns a copy of the tensor data.
# Use `tensor()` in order to get a pointer to the tensor.
output_data = interpreter.get_tensor(output_details[0]['index'])
print("Scores:\n Car: {:.2f}, Negative: {:.2f}, Person: {:.2f}".format(*output_data.flatten().tolist()))
```

    Scores:
     Car: 127.00, Negative: -128.00, Person: 127.00
    

 

And evaluate on the whole dataset


```python
def to_tflite_input_format(img):
    # Transform each image to the microcontroller raw capture format
    non_norm_img = (img*127.5)-0.5 # range [-128, 127]
    resized_img = tf.image.resize(non_norm_img, uc_img_size)
    uint8_img = tf.cast(resized_img, tf.int8)
    return uint8_img
```


```python
tflite_input_train_ds = train_ds.unbatch().map(lambda x, y: (to_tflite_input_format(x), y)).batch(1)
tflite_input_val_ds = val_ds.unbatch().map(lambda x, y: (to_tflite_input_format(x), y)).batch(1)
```


```python
def evaluate_tflite_model(interpreter, dataset, metrics): # This will take a while
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    
    for i, (img_batch, label_batch) in enumerate(dataset):
        interpreter.set_tensor(input_details['index'], img_batch)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details['index'])
        for metric in metrics:
            output_data_norm = tf.reshape((output_data+128)/255, [1, len(class_names)])
            metric.update_state(label_batch, output_data_norm)
        print("Step {} \r".format(i), end="")
        
    for metric in metrics:
        print("\n{}: {:.3f}".format(metric.name, metric.result()))
        
    return metrics
```


```python
METRICS = [
    tf.keras.metrics.BinaryAccuracy(name="bin_accuracy"),
    tf.keras.metrics.Precision(name="precision"),
    tf.keras.metrics.Recall(name="recall"),
    tfa.metrics.F1Score(len(class_names), average="macro", name="MacroF1"),
]

evaluate_tflite_model(interpreter, tflite_input_val_ds, METRICS)
```

    Step 3268 
    bin_accuracy: 0.884
    
    precision: 0.832
    
    recall: 0.825
    
    MacroF1: 0.829
    




    [<tensorflow.python.keras.metrics.BinaryAccuracy at 0x181056f0ca0>,
     <tensorflow.python.keras.metrics.Precision at 0x181056dc460>,
     <tensorflow.python.keras.metrics.Recall at 0x181056dc9a0>,
     <tensorflow_addons.metrics.f_scores.F1Score at 0x18103067e20>]




```python
METRICS[0].result()
```




    <tf.Tensor: shape=(), dtype=float32, numpy=0.88396055>



  
Good, now to get our tflite model to a readable C array, we can use `xxd`:

`$ xxd -i uc_final_model.tflite uc_final_model.h`

This creates `uc_model.h`, which looks exactly how we wanted:

```c
unsigned char uc_model_tflite[] = {
  0x20, 0x00, 0x00, 0x00, 0x54, ...};

unsigned int uc_model_tflite_len = 352768;
```

As a side note, remember to add the `const` modifier later. Finally, I would like to export our test image as well to a C readable format


```python
def dump_to_h_file(img):
    img = tf.reshape(img, (-1))
    out = "const int8_t test_img[{}] = {{ \n".format(img.shape[0])
    for i in range(img.shape[0]): # there are better ways
        out += str(img.numpy()[i]) + ","
        if (i+1) % 12 == 0:
            out+= "\n"
    out += "};"
    
    with open("test_image.h", "wt") as f:
        f.write(out)
```


```python
dump_to_h_file(tf.squeeze(input_data))
```

We can now continue by developing the C code.

# Deployment

For the full code please refer to the repository, but here is a small sample of the code that does the crucial part:

```c++
if (new_capture){
		new_capture=0;

		// Display inference information
		ILI9341_Draw_Image_From_OV7670((unsigned char*) frame_buffer, OV7670_QVGA_HEIGHT, OV7670_QVGA_WIDTH);
		ILI9341_Set_Rotation(SCREEN_HORIZONTAL_2);
		if (person_score >-50) ILI9341_Draw_Text("PERSON", 180, 210, GREEN, 2, BLACK);
		else ILI9341_Draw_Text("PERSON", 180, 210, RED, 2, BLACK);
		if (car_score >-50) ILI9341_Draw_Text("CAR", 80, 210, GREEN, 2, BLACK);
		else ILI9341_Draw_Text("CAR", 80, 210, RED, 2, BLACK);

		// TENSORFLOW
		// Fill input buffer
		uint16_t *pixel_pointer = (uint16_t *)frame_buffer;
		uint16_t input_ix = 0;
		for (uint32_t pix=0; pix<OV7670_QVGA_HEIGHT*OV7670_QVGA_WIDTH; pix++){
			// Convert from RGB55 to RGB888 and int8 range
			uint16_t color = pixel_pointer[pix];
			int16_t r = ((color & 0xF800) >> 11)*255/0x1F - 127;
			int16_t g = ((color & 0x07E0) >> 5)*255/0x3F - 127;
			int16_t b = ((color & 0x001F) >> 0)*255/0x1F - 127;

			model_input->data.int8[input_ix] = r;
			model_input->data.int8[input_ix+1] = g;
			model_input->data.int8[input_ix+2] = b;

			input_ix += 3;
		}

		// Run inference, measure time and report any error
		timestamp = htim->Instance->CNT;
		TfLiteStatus invoke_status = interpreter->Invoke();
		if (invoke_status != kTfLiteOk) {
			TF_LITE_REPORT_ERROR(error_reporter, "Invoke failed");
			return;
		}
		timestamp = htim->Instance->CNT - timestamp;
		car_score = model_output->data.int8[0];
		neg_score = model_output->data.int8[1];
		person_score = model_output->data.int8[2];
		// END TENSORFLOW

		// Print inference info
		buf_len = sprintf(buf,
						"car: %+*d, neg: %+*d, person: %+*d | Duration: %lu ms\r\n",
						4,car_score, 4,neg_score, 4,person_score , timestamp/1000);
				HAL_UART_Transmit(huart, (uint8_t *)buf, buf_len, 100);

		// Capture a new image
		ov7670_startCap(OV7670_CAP_SINGLE_FRAME, (uint32_t)frame_buffer);

	}
```

When a new image is captured by the camera, we display the image and the predictions of the previous one(more on this in just a few lines). Then we fill the model input buffer transforming the RGB565 frame buffer to RGB888 with the same logic we have applied before. After that, we run inference and print the predictions through UART.

The fact that the the predictions are one image delayed could be easily solved by displaying the image after inference. Unfortunately, the tensor arena buffer needed by tflite micro fills too much RAM and the frame buffer overflows it(there is not enough contiguous RAM). To solve this, I did a little trick: I oversized the tensor arena and used some of that buffer to store the frame buffer. That means that for the image to be displayed the `Invoke()` function should not have been called yet or the frame buffer would be corrupted. 

On an STM32H743, inference takes 1195 ms which is more than enough for the application we have been building.

(TODO further tests and stats)

# References

- [Coco dataset](https://cocodataset.org/#home)
- [CocoAPI for Python3 and Windows](https://github.com/philferriere/cocoapi#egg=pycocotools^&subdirectory=PythonAPI)
- [Visual wake word dataset](https://arxiv.org/abs/1906.05721)
- [Python docs](https://docs.python.org/3/)
- A lot of [stackoverflow](https://stackoverflow.com/)
- [The Unknown Benefits of using a Soft-F1 Loss in Classification Systems](https://towardsdatascience.com/the-unknown-benefits-of-using-a-soft-f1-loss-in-classification-systems-753902c0105d)
- [8 Tactics to Combat Imbalanced Classes in Your Machine Learning Dataset](https://machinelearningmastery.com/tactics-to-combat-imbalanced-classes-in-your-machine-learning-dataset/)
- [Classification on imbalanced data](https://www.tensorflow.org/tutorials/structured_data/imbalanced_data)
- [Handling Data Imbalance in Multi-label Classification (MLSMOTE)](https://medium.com/thecyphy/handling-data-imbalance-in-multi-label-classification-mlsmote-531155416b87)
- [Tensorflow guides](https://www.tensorflow.org/guide) and [tensorflow tutorials](https://www.tensorflow.org/tutorials)
- [EfficentNet](https://arxiv.org/abs/1905.11946)
- [NASNet](https://arxiv.org/abs/1707.07012)
- [The Unknown Benefits of using a Soft-F1 Loss in Classification Systems](https://towardsdatascience.com/the-unknown-benefits-of-using-a-soft-f1-loss-in-classification-systems-753902c0105d)
- [Cars dataset](https://ai.stanford.edu/~jkrause/cars/car_dataset.html)



```python

```
