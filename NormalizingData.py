import json
from uuid import uuid4
def normalize_faq():
    with open("WE_FAQ_data.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    normalized_docs = []
    for category, qa_pairs in data.items():
        for question, answer in qa_pairs.items():
            text = (f"Category: {category}\n\n"f"Question: {question}\n\n"f"Answer: {answer}")
            document = {
                "id": str(uuid4()),

                "text": text,

                "metadata": {

                    "type": "faq",
                    "category": category
                }
            }
            normalized_docs.append(document)

    with open("normalized_faq.json","w",encoding="utf-8") as f:
        json.dump(normalized_docs,f,ensure_ascii=False,indent=4)

def normalize_branches():
    with open("WE_Branches_data.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    normalized_docs = []
    governorates = data.get("governorates", {})

    for governorate_name, governorate_data in governorates.items():
        districts = governorate_data.get("districts", {})
        for district_name, branches in districts.items():
            for branch in branches:
                text = (
                    f"Governorate: {governorate_name}\n\n"
                    f"District: {district_name}\n\n"
                    f"Branch Name: {branch.get('name', '')}\n\n"
                    f"Address: {branch.get('address', '')}\n\n"
                    f"Working Hours: {branch.get('work_time', '')}"
                )
                document = {
                    "id": str(uuid4()),
                    "text": text,

                    "metadata": {

                        "type": "branch",

                        "governorate": governorate_name,

                        "district": district_name
                    }
                }
                normalized_docs.append(document)

    with open("normalized_branches.json","w",encoding="utf-8") as f:
        json.dump(normalized_docs,f,ensure_ascii=False,indent=4)

normalize_faq()
normalize_branches()


