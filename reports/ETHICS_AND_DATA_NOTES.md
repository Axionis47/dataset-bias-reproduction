# Ethics and Data Usage Notes

## Dataset Licences and Terms of Service

This reproduction project uses publicly available datasets with appropriate licences. We strictly adhere to all terms of service and ethical guidelines.

### Dataset Licence Summary

| Dataset | Licence | Commercial Use | Attribution Required | Notes |
|---------|---------|----------------|---------------------|-------|
| **YFCC100M** | CC BY 2.0 | ✅ Yes | ✅ Yes | Flickr user-uploaded content |
| **CC12M** | Various | ⚠️ Mixed | ✅ Yes | Web-crawled, check individual images |
| **DataComp-1B** | Various | ⚠️ Mixed | ✅ Yes | Common Crawl derived |
| **WIT** | CC BY-SA 3.0 | ✅ Yes | ✅ Yes | Wikipedia content |
| **LAION-2B** | Various | ⚠️ Mixed | ✅ Yes | Web-crawled, filtered |
| **ImageNet-1K** | Custom | ❌ Research Only | ✅ Yes | Requires registration |

### Ethical Considerations

#### Data Collection Ethics
- **Consent:** We use only datasets where images were made publicly available by their creators
- **Privacy:** No personally identifiable information is extracted or stored
- **Bias Awareness:** This study explicitly examines dataset bias as a research topic
- **Fair Use:** Usage falls under academic research and educational purposes

#### Potential Biases and Limitations
- **Geographic Bias:** Datasets may over-represent certain regions (primarily Western content)
- **Demographic Bias:** User-generated content may not represent global demographics equally
- **Temporal Bias:** Images reflect the time periods when datasets were collected
- **Platform Bias:** Each source platform has its own user base and content characteristics

#### Mitigation Strategies
- **Transparency:** All biases and limitations are documented and reported
- **Reproducibility:** Complete methodology and code are made available
- **Academic Purpose:** Research aims to understand and quantify bias, not exploit it
- **Responsible Reporting:** Results include discussion of limitations and ethical implications

### Dataset-Specific Notes

#### YFCC100M (Yahoo Flickr Creative Commons 100M)
- **Source:** Flickr user uploads with Creative Commons licences
- **Ethical Status:** ✅ Approved for research use
- **Bias Considerations:** Photographer demographics, geographic distribution
- **Access:** Publicly available, no registration required

#### CC12M (Conceptual Captions 12M)
- **Source:** Web-crawled image-text pairs
- **Ethical Status:** ✅ Approved for research use
- **Bias Considerations:** Web content bias, language bias (primarily English)
- **Access:** Publicly available through Google Research

#### DataComp-1B
- **Source:** Common Crawl web data, filtered and processed
- **Ethical Status:** ✅ Approved for research use
- **Bias Considerations:** Web content bias, filtering algorithm bias
- **Access:** Publicly available through research consortium

#### WIT (Wikipedia Image Text)
- **Source:** Wikipedia articles and associated images
- **Ethical Status:** ✅ Approved for research use
- **Bias Considerations:** Wikipedia editor demographics, topic coverage bias
- **Access:** Publicly available under CC BY-SA licence

#### LAION-2B (Large-scale Artificial Intelligence Open Network)
- **Source:** Web-crawled data, CLIP-filtered
- **Ethical Status:** ✅ Approved for research use
- **Bias Considerations:** CLIP model bias, web content bias
- **Access:** Publicly available, URLs and metadata provided

#### ImageNet-1K
- **Source:** Search engine results, manually curated
- **Ethical Status:** ⚠️ Requires registration and agreement
- **Bias Considerations:** Search engine bias, manual curation bias
- **Access:** Registration required, research use only
- **Note:** If access cannot be obtained, this dataset will be marked as SKIPPED

### Data Processing Ethics

#### Privacy Protection
- **No Face Recognition:** We do not perform facial recognition or identification
- **No Personal Data Extraction:** Only image content is used for classification
- **Anonymised Processing:** No user identifiers or personal information is stored

#### Content Filtering
- **Harmful Content:** We rely on dataset providers' content filtering
- **Copyright Respect:** Only use images with appropriate licences
- **Takedown Compliance:** Will remove content upon legitimate takedown requests

### Reproducibility and Transparency

#### Open Science Principles
- **Code Availability:** Complete source code is publicly available
- **Data Transparency:** All data sources and processing steps are documented
- **Result Reproducibility:** Experiments can be reproduced with provided instructions
- **Methodology Disclosure:** All experimental details are fully documented

#### Responsible AI Practices
- **Bias Documentation:** All known biases are explicitly documented
- **Limitation Acknowledgment:** Clear discussion of study limitations
- **Ethical Review:** Regular review of ethical implications
- **Community Engagement:** Open to feedback and ethical concerns

### Contact and Compliance

For questions about data usage, ethical concerns, or takedown requests:
- **Repository:** https://github.com/Axionis47/dataset-bias-reproduction
- **Issues:** https://github.com/Axionis47/dataset-bias-reproduction/issues
- **Email:** [Contact information to be added]

### Compliance Checklist

- [x] All datasets have appropriate licences for research use
- [x] Terms of service are respected for all data sources
- [x] No personally identifiable information is collected or stored
- [x] Bias and limitations are transparently documented
- [x] Code and methodology are made publicly available
- [x] Ethical considerations are regularly reviewed
- [ ] ImageNet access agreement obtained (pending)
- [ ] Final ethical review completed before publication

### Updates and Revisions

This document will be updated as the project progresses and new ethical considerations arise.

**Last Updated:** 2024-12-19  
**Version:** 1.0  
**Next Review:** Upon completion of data collection phase
