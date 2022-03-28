## INSTALLATION
1. Add "catalog.csv", "inverse.csv", "hierarchy.json", "inverse_attribute_values.csv" in temp directory.
- catalog.csv: Flipkart catalog formate csv containing "pid","title","brand","max_price","min_price","best_price","url","image_url" columns
- inverse.csv: csv containing mapping from root words to all different forms of words present in catalog.csv
- "hierarchy.json": hierarchy of categories and corresponding attributes
- "inverse_attribute_values.csv": csv containing mapping from tags to their different forms
2. run command `python -m flask run` in terminal

## SEARCH APPROACH DESCRIPTION

0. score is calculated for each query-catalog item pair and top scoring catalog items are displayed.
1. query words can contain all tags present in hierarchy.json, their different forms as present in inverse_attribute_values.csv and their root word as per inverse.csv.
2. sub-categories are divided into two parts, "common garments", "category specific_garments".
- category specific: those which belong to only one of the categories. e.g saree->female, vest->male
- common: those which could fall under any category. e.g shirt, pant, t-shirt
category specific garments are given more weights compared to common garments while computing scores.
3. attribute values are given lower weights compared to garment. e.g floral has lower weight compared to top 
