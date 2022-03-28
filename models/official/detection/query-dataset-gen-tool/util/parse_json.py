import json
import pdb

def parse_json(file_name="temp/hierarchy.json"):
    file = open(file_name)
    data = json.load(file)

    categories = []
    items = {}
    attributes = {}

    for category in data["level0"]:
        category_name = data["level0"][category]
        categories.append(category_name)

        items.update({category_name: []})

        for item in data["level1"][category]:
            item_name = data["level1"][category][item]
            items[category_name].append(item_name)

            attributes.update({category_name + "_" + item_name: {}})

            for attribute in data["level2"][category][item]:
                attribute_name = data["level2"][category][item][attribute]

                attributes[category_name + "_" + item_name].update({attribute_name: []})

                for attribute_value in data["level3"][category][item][attribute]:
                    attribute_value = data["level3"][category][item][attribute][attribute_value]

                    attributes[category_name + "_" + item_name][attribute_name].append(attribute_value)
    return categories, items, attributes
