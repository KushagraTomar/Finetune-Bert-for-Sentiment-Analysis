# 📚 BERT Sentiment Analysis Fine-Tuning

This ipynb file demonstrates how to fine-tune a pre-trained **BERT** model (`bert-base-uncased`) for **binary sentiment classification** on the **IMDb movie reviews dataset** (50,000 labeled samples).

It covers:
- ✅ Data preprocessing & tokenization
- ✅ PyTorch DataLoaders
- ✅ Fine-tuning with `transformers` & `torch`
- ✅ Training loss tracking & visualization
- ✅ Model evaluation with detailed metrics (Accuracy, ROC-AUC, PR-AUC, MCC, Cohen’s Kappa, Confusion Matrix)

---

## 🗂️ Dataset

- **IMDb** dataset from [Hugging Face Datasets](https://huggingface.co/datasets/stanfordnlp/imdb)
  - 25,000 training samples
  - 25,000 test samples
  - Labels: `0` (negative) & `1` (positive)

---

## 🚀 Model

- **Pre-trained:** `bert-base-uncased` (Hugging Face Transformers)
- **Task:** Sequence classification (`BertForSequenceClassification`)
- **Max sequence length:** 256 tokens


---

## 🎯 Why This Project Matters

- Demonstrates practical **transfer learning** with modern NLP.
- Shows how to adapt BERT for a classic real-world NLP task.
- Includes a clear training and evaluation pipeline that can be reused for other text classification tasks.
- Provides reproducible results and multiple ways to measure model quality.
- Helps beginners and intermediate ML practitioners learn fine-tuning workflows using Hugging Face Transformers and PyTorch.

---

## 💡 Key Takeaways

- Fine-tuning pre-trained models drastically reduces the amount of data and time needed for strong NLP results.
- Proper evaluation and visualization are crucial for trusting a model’s predictions.
- This project can be adapted for other datasets (e.g., Twitter sentiment, product reviews) with minimal changes.

---

## 🤝 Contributing

Open to pull requests, suggestions, or improvements for extending this baseline to other NLP tasks!

---

## 🙋‍♂️ Author

Made with ❤️ using PyTorch & 🤗 Transformers.

---

## 📌 References

- [Hugging Face Transformers](https://github.com/huggingface/transformers)
- [Hugging Face Datasets](https://github.com/huggingface/datasets)
- [IMDb Dataset](https://huggingface.co/datasets/stanfordnlp/imdb)
