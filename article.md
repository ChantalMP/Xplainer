
We propose a new way of explainability for zero-shot diagnosis prediction in the clinical domain. Instead of directly predicting a diagnosis, we prompt the model to classify the existence of descriptive observations, which a radiologist would look for on an X-Ray scan, and use the descriptor probabilities to estimate the likelihood of a diagnosis, making our model explainable by design. For this we leverage BioVil, a pretrained CLIP model for X-rays and apply contrastive observation-based prompting. We evaluate Xplainer on two chest X-ray
datasets, CheXpert and ChestX-ray14, and demonstrate its effectiveness
in improving the performance and explainability of zero-shot diagnosis.
**Authors**: [Chantal Pellegrini][cp], [Matthias Keicher][mk], [Ege Özsoy][eo], [Petra Jiraskova][pj], [Rickmer Braren][rb], [Nassir Navab][nn]

[cp]:https://www.cs.cit.tum.de/camp/members/chantal-pellegrini/
[eo]:https://www.cs.cit.tum.de/camp/members/ege-oezsoy/
[mk]:https://www.cs.cit.tum.de/camp/members/matthias-keicher/
[pj]:https://campus.tum.de/tumonline/ee/ui/ca2/app/desktop/#/pl/ui/$ctx/visitenkarte.show_vcard?$ctx=design=ca2;header=max;lang=de&pPersonenGruppe=3&pPersonenId=46F3A857F258DEE6
[rb]:https://radiologie.mri.tum.de/de/person/prof-dr-rickmer-f-braren
[nn]:https://www.cs.cit.tum.de/camp/members/cv-nassir-navab/nassir-navab/

**License**: MIT

**Where to send questions or comments about the model**: Open an issue on [`Xplainer`](https://github.com/ChantalMP/Xplainer) repo.

**Intended Use**: This model is intended to be used solely for (I) future research on visual-language processing and (II) reproducibility of the experimental results reported in the reference paper.

**Primary intended uses/users**: Vision-Language and CAD researchers 


## Citation
```bib
@article{pellegrini2023xplainer,
  title={Xplainer: From X-Ray Observations to Explainable Zero-Shot Diagnosis},
  author={Pellegrini, Chantal and Keicher, Matthias and {\"O}zsoy, Ege and Jiraskova, Petra and Braren, Rickmer and Navab, Nassir},
  journal={arXiv preprint arXiv:2303.13391},
  year={2023}
}
```