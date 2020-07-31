import gc
import json
import os

import pandas as pd
import visual_genome.utils as utils
from nltk.corpus import wordnet as wn
from tqdm import tqdm
from visual_genome.models import (Image, Object, Attribute, Relationship,
                                  Graph, Synset)

image_specific_attributes = {
    "shape_size",
    "colour_patterns",
    "texture_material",
    "structure"
}

if not os.path.exists("cache"):
    os.makedirs("cache")

SIMILARITY_CACHE_PATH = ".cache/similarity_cache.json"

if os.path.exists(SIMILARITY_CACHE_PATH):
    print("Loading similarity cache...")
    try:
        with open(SIMILARITY_CACHE_PATH, "r") as in_file:
            similarity_cache = json.load(in_file)
    except:
        print("Corrupted cache file...recreating!")
        similarity_cache = {}
    print("Loaded {} similarities".format(len(similarity_cache)))
else:
    similarity_cache = {}


def get_all_image_data(data_dir=None, as_dict=False):
    """
    Get Image ids from start_index to end_index.
    """
    if data_dir is None:
        data_dir = utils.get_data_dir()
    dataFile = os.path.join(data_dir, 'image_data.json')
    data = json.load(open(dataFile))
    if not as_dict:
        return [utils.parse_image_data(image) for image in data]

    return {
        image['id'] if 'id' in image else image['image_id']: utils.parse_image_data(image) for image in data
    }


def get_region_descriptions(image_ids, data_dir=None):
    """
    Get all region descriptions.
    """
    if data_dir is None:
        data_dir = utils.get_data_dir()
    data_file = os.path.join(data_dir, 'region_descriptions.json')
    image_data = get_all_image_data(data_dir, True)

    images = json.load(open(data_file))
    output = []
    for image in images:
        if image["id"] in image_ids and len(image["regions"]) > 0:
            output.append(utils.parse_region_descriptions(
                image['regions'], image_data[image['id']]))
    return output


def get_all_region_descriptions(data_dir=None):
    """
    Get all region descriptions.
    """
    if data_dir is None:
        data_dir = utils.get_data_dir()
    data_file = os.path.join(data_dir, 'region_descriptions.json')
    image_data = get_all_image_data(data_dir)
    image_map = {}
    for d in image_data:
        image_map[d.id] = d
    images = json.load(open(data_file))
    output = []
    for image in images:
        output.append(utils.parse_region_descriptions(
            image['regions'], image_map[image['id']]))
    return output


def get_all_qas(data_dir=None):
    """
    Get all question answers.
    """
    if data_dir is None:
        data_dir = utils.get_data_dir()
    data_file = os.path.join(data_dir, 'question_answers.json')
    image_data = get_all_image_data(data_dir)
    image_map = {}
    for d in image_data:
        image_map[d.id] = d
    images = json.load(open(data_file))
    output = []
    for image in images:
        output.append(utils.parse_QA(image['qas'], image_map))
    return output


# --------------------------------------------------------------------------------------------------
# get_scene_graphs and sub-methods


def get_scene_graph(image_id, images='data/',
                    image_data_dir='data/by-id/',
                    synset_file='data/synsets.json'):
    """
    Load a single scene graph from a .json file.
    """
    if type(images) is str:
        # Instead of a string, we can pass this dict as the argument `images`
        # Instead of a string, we can pass this dict as the argument `images`
        images = {img.id: img for img in get_all_image_data(images)}

    fname = str(image_id) + '.json'
    image = images[image_id]
    if not os.path.exists(image_data_dir + fname):
        return None
    data = json.load(open(image_data_dir + fname, 'r'))

    scene_graph = parse_graph_local(data, image)
    scene_graph = init_synsets(scene_graph, synset_file)
    return scene_graph


def get_scene_graphs(image_ids,
                     data_dir='data/', image_data_dir='data/by-id/',
                     min_rels=0, max_rels=100):
    """
    Get scene graphs given locally stored .json files;
    requires `save_scene_graphs_by_id`.

    image_indexes: list of image id as saved by `save_scene_graphs_by_id`
    data_dir : directory with `image_data.json` and `synsets.json`
    image_data_dir : directory of scene graph jsons saved by image id
                   (see `save_scene_graphs_by_id`)
    min_rels, max_rels: only get scene graphs with at least / less
                      than this number of relationships
    """
    images = {img.id: img for img in get_all_image_data(data_dir)}
    scene_graphs = []

    for image_id in tqdm(image_ids):
        scene_graph = get_scene_graph(
            image_id, images, image_data_dir, data_dir + 'synsets.json')

        if scene_graph:
            n_rels = len(scene_graph.relationships)
            if (min_rels <= n_rels <= max_rels):
                scene_graphs.append(scene_graph)

    return scene_graphs


def map_object(object_map, obj):
    """
    Use object ids as hashes to `visual_genome.models.Object` instances.
    If item not in table, create new `Object`. Used when building
    scene graphs from json.
    """

    oid = "gw_{}".format(obj['object_id']) if obj.get("guesswhat", False) else obj["object_id"]
    obj['id'] = oid
    del obj['object_id']

    if oid in object_map:
        object_ = object_map[oid]

    else:
        if 'attributes' in obj:
            attrs = obj['attributes']
            del obj['attributes']
        else:
            attrs = []

        if 'abstract_attributes' in obj:
            abs_attrs = obj['abstract_attributes']
            del obj['abstract_attributes']
        else:
            abs_attrs = []

        if 'situated_attributes' in obj:
            sit_attrs = obj['situated_attributes']
            del obj['situated_attributes']
        else:
            sit_attrs = []

        if 'w' in obj:
            obj['width'] = obj['w']
            obj['height'] = obj['h']
            del obj['w'], obj['h']

        if 'guesswhat' in obj:
            obj['guesswhat'] = True
        else:
            obj['guesswhat'] = False

        object_ = Object(**obj)

        object_.attributes = attrs
        object_.abstract_attributes = abs_attrs
        object_.situated_attributes = sit_attrs
        object_map[oid] = object_

    return object_map, object_


global count_skips
count_skips = [0, 0]


def parse_graph_local(data, image, verbose=False):
    """
    Modified version of `utils.ParseGraph`.
    """
    global count_skips
    objects = []
    object_map = {}
    relationships = []
    attributes = []

    for obj in data['objects']:
        object_map, o_ = map_object(object_map, obj)
        objects.append(o_)
    for rel in data['relationships']:
        if rel['subject_id'] in object_map and rel['object_id'] in object_map:
            object_map, s = map_object(
                object_map, {'object_id': rel['subject_id']})
            v = rel['predicate']
            object_map, o = map_object(
                object_map, {'object_id': rel['object_id']})
            rid = rel['relationship_id']
            relationships.append(Relationship(rid, s, v, o, rel['synsets']))
        else:
            # Skip this relationship if we don't have the subject and object in
            # the object_map for this scene graph. Some data is missing in this
            # way.
            count_skips[0] += 1
    if 'attributes' in data:
        for attr in data['attributes']:
            a = attr['attribute']
            if a['object_id'] in object_map:
                attributes.append(Attribute(attr['attribute_id'],
                                            Object(a['object_id'], a['x'],
                                                   a['y'], a['w'], a['h'],
                                                   a['names'], a['synsets']),
                                            a['attributes'], a['synsets']))
            else:
                count_skips[1] += 1
    if verbose:
        print('Skipped {} rels, {} attrs total'.format(*count_skips))
    return Graph(image, objects, relationships, attributes)


def init_synsets(scene_graph, synset_file):
    """
    Convert synsets in a scene graph from strings to Synset objects.
    """
    syn_data = json.load(open(synset_file, 'r'))
    syn_class = {s['synset_name']: Synset(
        s['synset_name'], s['synset_definition']) for s in syn_data}

    for obj in scene_graph.objects:
        new_obj_synsets = []

        for sn in obj.synsets:
            if isinstance(sn, Synset):
                new_obj_synsets.append(sn)
            elif isinstance(sn, str):
                if sn in syn_class:
                    new_obj_synsets.append(syn_class[sn])
                else:
                    new_obj_synsets.append(Synset(sn, wn.synset(sn)))

        obj.synsets = new_obj_synsets

    for rel in scene_graph.relationships:
        rel.synset = [syn_class[sn] for sn in rel.synset]
    for attr in scene_graph.attributes:
        attr.synset = [syn_class[sn] for sn in attr.synset]

    return scene_graph


def extract_category_attributes(category_attributes):
    abstract_attributes = []

    for cat, cat_attrs in category_attributes.items():
        for att in cat_attrs:
            norm_att = att.replace("beh_-_", "_").replace("_", " ")

            abstract_attributes.append(
                norm_att
            )

    return abstract_attributes


def format_box(bbox):
    return [
        bbox[0],
        bbox[1],
        bbox[2] + bbox[0],
        bbox[3] + bbox[1]
    ]


def extract_positional_attributes(image, bbox):
    # we first compute the center of the image as bounding box
    box_width = image.width / 4
    box_height = image.height / 4

    center_box = [
        box_height,
        box_height,
        box_width + box_height,
        box_width + box_height
    ]

    # convert to format X1, Y1, X2, Y2
    boxA = format_box(bbox)
    boxB = format_box(center_box)

    positional_attributes = []

    if boxA[0] > boxB[2]:
        # boxA is right of boxB
        positional_attributes.append("right_image")
    elif boxB[0] > boxA[2]:
        # boxA is left of boxB
        positional_attributes.append("left_image")

    if boxA[3] < boxB[1]:
        # boxA is above boxB
        positional_attributes.append("top_image")
    elif boxA[1] > boxB[3]:
        # boxA is below boxB
        positional_attributes.append("bottom_image")

    if not positional_attributes:
        positional_attributes.append("center")

    return positional_attributes


def init_attributes(scene_graph, vg_image, attributes_data, gw_image_data):
    """
    Convert synsets in a scene graph from strings to Synset objects.
    """

    for obj in scene_graph["objects"]:
        if "attributes" not in obj:
            obj["situated_attributes"] = []
            obj["abstract_attributes"] = []
        else:
            obj["situated_attributes"] = obj["attributes"]
            obj["abstract_attributes"] = []

        # extract positional attributes from the object bbox

        obj["situated_attributes"].extend(
            extract_positional_attributes(vg_image, [obj["x"], obj["y"], obj["w"], obj["h"]]))

        if "synsets" in obj and obj["synsets"]:
            category_attr = attributes_data[attributes_data["wordnet_id"] == obj["synsets"][0]]

            # check if we can map the current object to the VISA dataset via Wordnet synsets
            if not category_attr.empty:
                attributes = category_attr["data"].values[0]["attributes"]
                types = category_attr["data"].values[0]["types"]
                obj["abstract_attributes"].extend(extract_category_attributes(attributes))
                obj["abstract_attributes"].extend([t.replace(" ", "_") for t in types])
            else:
                # the current object has a synset not matching with any category, we try with a similarity based approach
                # current threshold is 0.75 and we use the WUP similarity measure
                object_syn = wn.synset(obj["synsets"][0])

                best_match = (None, 0)

                if object_syn.name() in similarity_cache:
                    best_match = (similarity_cache[object_syn.name()], None)
                else:
                    for category in attributes_data["wordnet_id"]:
                        category_syn = wn.synset(category)
                        similarity_score = object_syn.wup_similarity(category_syn)

                        if similarity_score >= 0.75 and best_match[1] < similarity_score:
                            best_match = (category_syn.name(), similarity_score)

                    if best_match[0] is not None:
                        similarity_cache[object_syn.name()] = best_match[0]

                if best_match[0]:
                    category_attr = attributes_data[attributes_data["wordnet_id"] == best_match[0]]
                    attributes = category_attr["data"].values[0]["attributes"]
                    types = category_attr["data"].values[0]["types"]
                    obj["abstract_attributes"].extend(extract_category_attributes(attributes))
                    obj["abstract_attributes"].extend([t.replace(" ", "_") for t in types])

        obj["attributes"] = obj["situated_attributes"] + obj["abstract_attributes"]

    if gw_image_data is not None:
        for obj in gw_image_data["gw_objects"]:
            category_attr = attributes_data[attributes_data["concept_id"] == obj["category"]]
            if not category_attr.empty:
                gw_object = {
                    "synsets": [category_attr["wordnet_id"].values[0]],
                    "x": obj["bbox"][0],
                    "y": obj["bbox"][1],
                    "w": obj["bbox"][2],
                    "h": obj["bbox"][3],
                    "names": [obj["category"]],
                    "object_id": obj["id"],
                    "abstract_attributes": [],
                    "situated_attributes": extract_positional_attributes(vg_image, obj["bbox"]),
                    "attributes": [],
                    "guesswhat": True
                }

                attributes = category_attr["data"].values[0]["attributes"]
                types = category_attr["data"].values[0]["types"]
                gw_object["abstract_attributes"].extend(extract_category_attributes(attributes))
                gw_object["abstract_attributes"].extend([t.replace(" ", "_") for t in types])
                gw_object["attributes"] = gw_object["abstract_attributes"]
                scene_graph["objects"].append(gw_object)

    return scene_graph


# --------------------------------------------------------------------------------------------------
# This is a pre-processing step that only needs to be executed once.
# You can download .jsons segmented with these methods from:
#     https://drive.google.com/file/d/0Bygumy5BKFtcQ1JrcFpyQWdaQWM


def save_scene_graphs_by_id(data_dir='data/', image_data_dir='data/by-id/'):
    """
    Save a separate .json file for each image id in `image_data_dir`.

    Notes
    -----
    - If we don't save .json's by id, `scene_graphs.json` is >6G in RAM
    - Separated .json files are ~1.1G on disk
    - Run `add_attrs_to_scene_graphs` before `parse_graph_local` will work
    - Attributes are only present in objects, and do not have synset info

    Each output .json has the following keys:
      - "id"
      - "objects"
      - "relationships"
    """
    if not os.path.exists(image_data_dir):
        os.mkdir(image_data_dir)

    attributes_data = pd.read_json(data_dir + "visa.jsonl", orient="records", lines=True)
    with open(data_dir + "gw_vg_mapping.json") as in_file:
        gw_vg_metadata = json.load(in_file)

    vg_image_data = get_all_image_data(data_dir, True)

    all_data = json.load(open(os.path.join(data_dir, 'scene_graphs.json')))

    pbar = tqdm()
    for sg_data in all_data:
        if sg_data["image_id"] in vg_image_data:
            vg_image = vg_image_data[sg_data["image_id"]]
            if vg_image.coco_id is not None:
                coco_id = str(vg_image.coco_id)
                if coco_id in gw_vg_metadata:
                    gw_image_data = gw_vg_metadata[coco_id]

                    sg_data = init_attributes(sg_data, vg_image, attributes_data, gw_image_data)
                    img_fname = str(sg_data['image_id']) + '.json'
                    with open(os.path.join(image_data_dir, img_fname), 'w') as f:
                        json.dump(sg_data, f)

                    pbar.update(1)
    del all_data
    gc.collect()  # clear memory

    with open(SIMILARITY_CACHE_PATH, mode="w") as out_file:
        json.dump(similarity_cache, out_file)


def add_attrs_to_scene_graphs(data_dir='data/'):
    """
    Add attributes to `scene_graph.json`, extracted from `attributes.json`.

    This also adds a unique id to each attribute, and separates individual
    attibutes for each object (these are grouped in `attributes.json`).
    """
    attr_data = json.load(open(os.path.join(data_dir, 'attributes.json')))
    with open(os.path.join(data_dir, 'scene_graphs.json')) as f:
        sg_dict = {sg['image_id']: sg for sg in json.load(f)}

    id_count = 0
    for img_attrs in attr_data:
        attrs = []
        for attribute in img_attrs['attributes']:
            a = img_attrs.copy()
            del a['attributes']
            a['attribute'] = attribute
            a['attribute_id'] = id_count
            attrs.append(a)
            id_count += 1
        iid = img_attrs['image_id']
        sg_dict[iid]['attributes'] = attrs

    with open(os.path.join(data_dir, 'scene_graphs.json'), 'w') as f:
        json.dump(list(sg_dict.values()), f)
    del attr_data, sg_dict
    gc.collect()


# --------------------------------------------------------------------------------------------------
# For info on VRD dataset, see:
#   http://cs.stanford.edu/people/ranjaykrishna/vrd/

def get_scene_graphs_VRD(json_file='data/vrd/json/test.json'):
    """
    Load VRD dataset into scene graph format.
    """
    scene_graphs = []
    with open(json_file, 'r') as f:
        D = json.load(f)

    scene_graphs = [parse_graph_VRD(d) for d in D]
    return scene_graphs


def parse_graph_VRD(d):
    image = Image(d['photo_id'], d['filename'], d[
        'width'], d['height'], '', '')

    id2obj = {}
    objs = []
    rels = []
    atrs = []

    for i, o in enumerate(d['objects']):
        b = o['bbox']
        obj = Object(i, b['x'], b['y'], b['w'], b['h'], o['names'], [])
        id2obj[i] = obj
        objs.append(obj)

        for j, a in enumerate(o['attributes']):
            atrs.append(Attribute(j, obj, a['attribute'], []))

    for i, r in enumerate(d['relationships']):
        s = id2obj[r['objects'][0]]
        o = id2obj[r['objects'][1]]
        v = r['relationship']
        rels.append(Relationship(i, s, v, o, []))

    return Graph(image, objs, rels, atrs)
