mkdir ./model
unzip ./data/zp_raw_data.zip
mv zp_data ./data
python data_builder.py
rm -r ./data/zp_data
