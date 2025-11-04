# 204466 การเรียนรู้เชิงลึก (Deep Learning)

## โครงการสุดท้าย (Final Project)

**ทำเดี่ยวหรือทำเป็นกลุ่มได้ แต่สมาชิกในกลุ่มต้องไม่เกิน 2 คน**

**กำหนดส่ง 4 พฤศจิกายน (ก่อนเที่ยงคืน)**

## สิ่งที่ต้องทำ
- เลือกปัญหาที่การแก้ปัญหาโดย deep learning เป็นทางเลือกที่เหมาะสม
- สร้างโมเดล deep learning เพื่อแก้ปัญหาดังกล่าวโดยใช้ framework ที่เราได้เรียนรู้มาอย่าง Google Colab และ PyTorch หรือ TensorFlow
- Train deep learning model ด้วยอัลกอริทึมและ dataset ที่เหมาะสม
- ประเมินสมรรถนะของ deep learning network กับการแก้ปัญหาที่เราได้เลือกมา
- สามารถนำโมเดลอื่นๆมาใช้งานร่วมด้วยได้ _แต่ต้องมีอย่างน้อยหนึ่งโมเดลที่มาจากมันสมองและความคิดของตัวเอง_
- เขียนรายงานเกี่ยวกับโปรเจคที่ทำ ดูรายละเอียดสิ่งที่ต้องอธิบายในหัวข้อ “สิ่งที่ต้องตอบในรายงาน”

## สิ่งที่ห้ามทำ
- ลอกโค้ด ลอกแนวคิด ลอกงานของคนอื่น มานำเสนอและรายงานส่งว่าเป็นของตัวเอง ทุกคนที่เกี่ยวข้องจะได้ F ในวิชานี้ถ้าทำผิดกฏข้อนี้

## สิ่งที่ต้องตอบในรายงาน (final_report.pdf)
- หัวข้อ final project
- หัวข้อนี้น่าสนใจอย่างไร ทำไมถึงเลือกหัวข้อนี้มาทำเป็น final project
- ทำไมหัวข้อนี้จึุงต้องใช้ deep learning ในการแก้ปัญหา เปรียบเทียบกับการแก้ปัญหานี้ด้วยวิธีอื่นๆ วิธี deep learning มีข้อเด่น ข้อด้อยอย่างไร
- อธิบายสถาปัตยกรรม deep learning ที่ใช้ (feedforward NN CNN RNN GAN หรือ VAE) วาดรูปแสดงจำนวนโหนด weight bias รวมถึงการเชื่อมต่อ และ activation function ต่างๆให้ชัดเจน
- อธิบายโค้ด PyTorch หรือ TensorFlow รวมไปถึงโค้ดส่วนอื่นๆที่ใช้ในการเชื่อมต่อกับโมเดลอื่นๆ อย่างชัดเจน ส่วนไหนจัดการกับข้อมูล ส่วนไหนสร้างโมเดล ส่วนไหน train ฯลฯ
  - ส่งลิงค์ที่นำไปสู่โค้ด PyTorch หรือ TensorFlow ที่ทุกคนเข้าถึงได้มาด้วย ควรจะเป็น Github public repo link
- อธิบายวิธีในการ train ตัว deep learning network ที่เลือกมาใช้ รวมไปถึงอธิบาย dataset ที่เกี่ยวข้องและแหล่งที่มา
- อธิบายการประเมิน (evaluate) model แสดงค่า loss จากการ train และ metric ที่เหมาะสมในการประเมิ เช่น accuracy precision หรือ recall
- อธิบายบทความอ้างอิงและงานที่เกี่ยวข้อง
- ถ้าทำเป็นกลุ่ม บอกงานที่แต่ละคนทำและให้สัดส่วนเป็นเปอร์เซ็นต์ของงานทั้งหมด

## สิ่งที่ต้องส่ง (ส่งมาใน Google Classroom)
- ไฟล์ final_report.pdf
- Github link ที่มีโค้ด PyTorch ที่เกี่ยวข้องอยู่
  - แปลง final_report.pdf ให้เป็นไฟล์ README.md และ commit ไฟล์นี้ลงใน Github repo เดียวกันนี้

## เกณฑ์การให้คะแนน
- คุณภาพของผลงาน (65%)
  - ปัญหาน่าสนใจ มีความยากและท้าทาย
  - แนวทางแก้ปัญหาเหมาะสมและชัดเจน
  - คุณภาพของโค้ด PyTorch การประเมิน และผลลัพธ์สุดท้ายที่ได้
- คุณภาพของรายงาน (35%)
  - การตอบคำถามชัดเจน ผู้่อ่านเข้าใจได้ง่าย มีภาพประกอบ

## ตัวอย่างหัวข้อ final project
- สร้างเมนูอาหาร fusion โดยใช้ RNN
- สร้างดนตรีแนวใหม่จาก RNN
- Fine-tune LLM ให้เป็นตัวแทนของตลกที่เราชื่นชอบ
- สร้างภาพหรือศิลปะของศิลปินที่เราสนใจโดย VAE หรือ GAN
- สร้างศิลปะลายไทยจากคำสั่งง่ายๆ เช่นลายกระหนก ลายกระจัง
- สร้างโมเดลทำนายแนวโน้มของราคาทรัพย์สินต่างๆ เช่น ทองคำ Bitcoin S&P500

## ตัวอย่าง public dataset ที่สามารถใช้ได้
1. Kaggle Datasets
- https://www.kaggle.com/datasets


2. UCI Machine Learning Repository
- https://archive.ics.uci.edu/ml/index.php


3. Google Dataset Search
- https://datasetsearch.research.google.com/


4. Hugging Face Datasets
- https://huggingface.co/datasets


5. Computer Vision Datasets
- ImageNet — http://image-net.org/ (large-scale)
- COCO — https://cocodataset.org/
- Open Images — https://storage.googleapis.com/openimages/web/index.html


6. Natural Language / Speech Datasets
- Common Crawl — massive web text corpus (https://commoncrawl.org/)
- LibriSpeech — audiobook speech data (http://www.openslr.org/12/)
- Tatoeba / WMT — translation datasets


7. Government & Open Data Portals
- data.gov (USA): https://www.data.gov/
- data.gov.uk, data.go.th, etc. for country-specific portals.
- Contain real-world public data: demographics, transportation, climate, health, etc.

## ตัวอย่าง public model ที่สามารถใช้ได้
1. Hugging Face Model Hub
https://huggingface.co/models

2. Torch Hub (PyTorch Models)
https://pytorch.org/hub/

3. NVIDIA NGC
- NVIDIA NGC Catalog — https://catalog.ngc.nvidia.com/models
  (optimized for GPUs; includes diffusion, LLMs, etc.)
