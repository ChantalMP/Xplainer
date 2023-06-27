from pathlib import Path
from PIL import Image
import csv
from tqdm import tqdm

root_dir = Path('ChestX-ray14') # change to path of the dataset


def convert_images():
    text_file_path = root_dir / 'test_list.txt'
    images_path = root_dir / 'images'
    export_jpg_images_path = root_dir / 'images_jpg'
    export_jpg_images_path.mkdir(exist_ok=True)

    with open(text_file_path, 'r') as f:
        lines = f.readlines()
        for line in tqdm(lines):
            image_name = line.strip()
            image_path = images_path / image_name
            # Convert image to jpg for reducing the size
            img = Image.open(image_path)
            img = img.convert('RGB')
            new_size = (512, int(512 * img.size[1] / img.size[0])) if img.size[0] < img.size[1] else (int(512 * img.size[0] / img.size[1]), 512)
            img = img.resize(new_size)  # resize smaller edge to 512 and keep aspect ratio
            image_name = image_name.replace('.png', '.jpg')
            image_path = export_jpg_images_path / image_name
            img.save(image_path)


def convert_labels():
    input_file = root_dir / 'Data_Entry_2017_v2020.csv'
    output_file = root_dir / 'Data_Entry_2017_v2020_modified.csv'

    diseases = ['No Finding', 'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation',
                'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']
    output_diseases = ['No Finding', 'Atelectasis', 'Cardiomegaly', 'Pleural Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax',
                       'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']

    # Create a dictionary to store the output data
    output_data = []

    # Open the input file and read its contents
    with open(input_file) as f:
        reader = csv.DictReader(f, delimiter=',')
        for row in reader:
            # Get the patient ID from the filename
            patient_id = row['Patient ID']
            image_path = row['Image Index']
            output = [image_path, patient_id] + [0] * len(diseases)

            # Update the output row with disease labels
            for finding in row['Finding Labels'].split('|'):
                if finding in diseases:
                    output[diseases.index(finding) + 2] = 1
                else:
                    print('Disease not found: {}'.format(finding))

            output_data.append(output)

    # Write the output file
    with open(output_file, 'w') as f:
        writer = csv.DictWriter(f, fieldnames=['Path', 'Patient ID'] + output_diseases, delimiter=',')
        writer.writeheader()
        for row in output_data:
            writer.writerow(dict(zip(['Path', 'Patient ID'] + output_diseases, row)))


if __name__ == '__main__':
    convert_images()
    convert_labels()
