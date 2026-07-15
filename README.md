# A Network Approach to Mapping Responsibility in Wildfire Risk Mitigation

Code accompanying:

> Kim, M., Raine, H., Radke, J., & González, M. C. *A Network Approach to
> Mapping Responsibility in Wildfire Risk Mitigation.*

## Background

As wildfire risk escalates in the wildland urban interface/intermix (WUI), creating defensible space clear of flammable fuels is important for wildfire risk mitigation and management. By definition, defensible space is constrained to the property parcel's boundaries and wildfire risk assessments generally focus on individual parcels, but this overlooks any risk in neighboring properties. If we imagine defensible spaces as buffers that cross property boundaries and overlap, neighboring properties may share wildfire risk in the overlapping spaces. However, homeowners may not recognize these overlapping regions and spillover effects of mitigating risk, jeopardizing their properties and, ultimately, the entire neighborhood. To address this issue, we propose a novel paradigm to rethink defensible space management by mapping the homeowner's responsibility to mitigate wildfire risk in overlapping defensible space buffers with respect to their neighbors who may share or owe responsibility to mitigation. First, we develop three new spatial metrics (Personal Responsibility (PR), Shared Responsibility (SR), and Owed Responsibility (OR)) for each homeowner, computed as the product of wildfire risk and the corresponding area of responsibility. Here, we use a fire behavior model to generate rate of spread (ROS) as a measure of wildfire hazard and compute each responsibility metric. Second, we build the metrics into spatial responsibility networks, modeled using all the houses in a given neighborhood with nodes characterized by average PR and with links weighted by average SR (or OR) that are directed to designate who is responsible between neighboring properties. In our study site, we find that SR networks are composed of multiple sub-networks while OR networks emerge as high-responsibility clusters. This information is important to inform homeowners about their individual responsibilities, identify properties with greater responsibility, and gather neighbors with interconnected responsibility. Subsequently, we simulate different mitigation strategies by iteratively removing network links and monitoring how the total responsibility may change in the neighborhood. We use three strategies (random, localized, and targeted) for link removal. Results demonstrate that the targeted strategy (removing links in descending order of highest responsibility) reduces the total responsibility most rapidly and fragments the network into smaller components, concentrating mitigation on the highest-responsibility links. Through this study, we provide a framework for wildfire risk management, which can be used with different metrics and for various neighborhood layouts. Ultimately, the spatial responsibility metrics and networks provide a scalable and spatially-explicit approach to map complex wildfire risk in the WUI, which can help inform defensible space inspections, guide efficient resource allocation, improve neighborhood-level planning, and empower individual homeowners to make more risk-informed decisions.


## Repository structure

```
src/
  utils.py        Helper functions
  compute_sr.py   Compute PR/SR/OR/TR
  geometry.py     Calculating geometry of whole neighborhood
  network.py      Building networks (NetworkX graphs) for SR and OR networks
  plot.py         Visualization functions of main figures

notebooks/
  main.ipynb   
  study_area.ipynb

data/
  bldgs.shp, parcels.shp  
  Outputs/ROS_nomit.tif
```

## Setup

```bash
pip install -r requirements.txt
```

The repo contains a study area (demonstration neighborhood) `data/` for `notebooks/main.ipynb`. 
To use different study area datasets, replace `data_path` in the notebook with your own
`bldgs.shp`/`parcels.shp` (and your own risk raster for `risk_path`).

## Citation

If you use this code, please cite the paper above.
