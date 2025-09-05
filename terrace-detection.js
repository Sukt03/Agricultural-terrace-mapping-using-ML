
// Terrace Detection Workflow - Google Earth Engine Implementation

var b2 = ee.Image("projects/autoterraces/assets/BAND2");  // Green
var b3 = ee.Image("projects/autoterraces/assets/BAND3");  // Red
var b4 = ee.Image("projects/autoterraces/assets/BAND4");  // NIR
var dem = ee.Image("USGS/SRTMGL1_003");

var geometry = ee.FeatureCollection("projects/autoterraces/assets/shapefiles");
var proj = b2.projection();

Map.centerObject(geometry, 13);
Map.addLayer(geometry.style({color: 'cyan', fillColor: '00000000'}), {}, 'AOI');


var enriched = geometry.map(function(feature) {
  var geom = feature.geometry();
  var label = feature.get('label');

  // NDVI
  var liss = b2.rename('green')
    .addBands(b3.rename('red'))
    .addBands(b4.rename('nir'))
    .clip(geom);
  var ndvi = liss.normalizedDifference(['nir', 'red']).rename('NDVI');

  // HSV color transform
  var rgbImage = b3.addBands(b2).addBands(b2).clip(geom);
  var hsv = rgbImage.unitScale(0, 3000).rgbToHsv();

  // Sobel edges
  var gray = rgbImage.reduce(ee.Reducer.mean());
  var sobel = gray.convolve(ee.Kernel.sobel());

  // Terrain slope
  var slope = ee.Terrain.slope(dem.clip(geom).reproject(proj, null, 30));

  // NDVI-based mask
  var ndviMask = ndvi.lt(0.3).selfMask();

  // HSV-based mask
  var hsvMask = hsv.select('hue').gte(0.48).and(hsv.select('hue').lte(0.52))
    .and(hsv.select('saturation').gte(0.05)).and(hsv.select('saturation').lte(0.4))
    .and(hsv.select('value').gte(0.02)).and(hsv.select('value').lte(0.12)).selfMask();

  // Slope mask
  var slopeMask = slope.gte(3).and(slope.lte(35)).selfMask();

  // Edge threshold (60th percentile)
  var edgeThreshold = sobel.reduceRegion({
    reducer: ee.Reducer.percentile([60]),
    geometry: geom,
    scale: 10,
    maxPixels: 1e8
  }).get('mean');
  var edgeMask = sobel.gt(ee.Number(edgeThreshold)).selfMask();

  // Reducer: mean, min/max, percentiles
  var reducer = ee.Reducer.mean()
    .combine(ee.Reducer.minMax(), '', true)
    .combine(ee.Reducer.percentile([5, 25, 50, 75, 95]), '', true);

  // Regional statistics
  var ndviStats = ndvi.reduceRegion({reducer: reducer, geometry: geom, scale: 10, maxPixels: 1e9});
  var hsvStats = hsv.reduceRegion({reducer: reducer, geometry: geom, scale: 10, maxPixels: 1e9});
  var slopeStats = slope.reduceRegion({reducer: reducer, geometry: geom, scale: 30, maxPixels: 1e9});
  var sobelStats = sobel.reduceRegion({reducer: reducer, geometry: geom, scale: 10, maxPixels: 1e9});

  // NDVI mask area & pixel count
  var ndviMaskAreaImg = ndviMask.unmask(0).multiply(ee.Image.pixelArea()).rename('ndvi_mask');
  var ndviPixelCount = ndviMask.reduceRegion({
    reducer: ee.Reducer.count(),
    geometry: geom,
    scale: 10,
    maxPixels: 1e9
  }).get('NDVI');
  var ndviArea = ndviMaskAreaImg.reduceRegion({
    reducer: ee.Reducer.sum(),
    geometry: geom,
    scale: 10,
    maxPixels: 1e9
  }).get('ndvi_mask');

  // Attach features
  return feature.set(ndviStats)
    .set(hsvStats)
    .set(slopeStats)
    .set(sobelStats)
    .set({
      'edge_threshold_60p': edgeThreshold,
      'ndvi_mask_pixel_count': ndviPixelCount,
      'ndvi_mask_area_m2': ndviArea
    });
});

// Export Feature Table

Export.table.toDrive({
  collection: enriched,
  description: 'Terrace_Stats_Export',
  fileNamePrefix: 'terrace_stats',
  fileFormat: 'CSV'
});


// Load Sentinel-2 composite
var s2_dess = ee.ImageCollection('COPERNICUS/S2_SR')
  .filterBounds(dessAOI)
  .filterDate('2021-01-01', '2021-12-31')
  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 10))
  .select(['B2', 'B3', 'B4', 'B8'])
  .median()
  .clip(dessAOI);

var dem_d = ee.Image('USGS/SRTMGL1_003').clip(dessAOI);

// Derived features
var ndvi_d = s2_dess.normalizedDifference(['B8', 'B4']).rename('NDVI');
var slope_d = ee.Terrain.slope(dem_d).rename('slope');

// Feature stack
var stack_d = s2_dess.select(['B2','B3','B4','B8'])
  .rename(['blue','green','red','nir'])
  .addBands(ndvi_d)
  .addBands(slope_d);

var bands = ['blue','green','red','nir','NDVI','slope'];

// Sampling and balancing
var samples = stack_d.addBands(labelImage).sample({
  region: dessAOI,
  scale: 30,
  numPixels: 5000,
  seed: 42,
  geometries: true,
  tileScale: 4
}).map(function(f) {
  return f.set('label', ee.Number(f.get('label')).toInt());
});

// Balance classes
var class0 = samples.filter(ee.Filter.eq('label', 0));
var class1 = samples.filter(ee.Filter.eq('label', 1));
var n1 = class1.size();
var class0Sample = class0.randomColumn('rand').sort('rand').limit(n1);
var trainSet = class0Sample.merge(class1);

// Train Random Forest classifier (probability output)
var trainedClassifier = ee.Classifier.smileRandomForest({
  numberOfTrees: 100,
}).train({
  features: trainSet,
  classProperty: 'label',
  inputProperties: bands
}).setOutputMode('PROBABILITY');


var s2_user = ee.ImageCollection('COPERNICUS/S2_SR')
  .filterBounds(userAOI)
  .filterDate('2021-01-01', '2021-12-31')
  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 10))
  .select(['B2','B3','B4','B8'])
  .median()
  .clip(userAOI);

var dem_u = ee.Image('USGS/SRTMGL1_003').clip(userAOI);
var ndvi_u = s2_user.normalizedDifference(['B8','B4']).rename('NDVI');
var slope_u = ee.Terrain.slope(dem_u).rename('slope');

var stack_u = s2_user.rename(['blue','green','red','nir'])
  .addBands(ndvi_u)
  .addBands(slope_u);

// Probability map
var terraceProb = stack_u.select(bands).classify(trainedClassifier).rename('terrace_prob');

var probThreshold = 0.81;
var terraceMask_81 = terraceProb.gt(probThreshold).rename('terrace_mask').uint8();

// Export terrace mask
Export.image.toDrive({
  image: terraceMask_81,
  description: 'terraceMask_gt_081',
  folder: 'GEE_Exports',
  fileNamePrefix: 'terraceMask_gt_081',
  region: userAOI.geometry(),
  scale: 10,
  maxPixels: 1e13,
  crs: 'EPSG:4326'
});


var solarAzimuth = 135;
var solarElevation = 45;
var hillshade = ee.Terrain.hillshade(dem, solarAzimuth, solarElevation);
var hillshadeNorm = hillshade.divide(255).rename("hillshade_norm");

// NIR normalization
var stats = lissIV.reduceRegion({
  reducer: ee.Reducer.percentile([1,99]),
  geometry: aoi,
  scale: 10,
  maxPixels: 1e13
});
var minNIR = ee.Number(stats.get('NIR_p1'));
var maxNIR = ee.Number(stats.get('NIR_p99'));
var nirNorm = lissIV.subtract(minNIR).divide(maxNIR.subtract(minNIR)).clamp(0,1).rename("nir_norm");

// Shadow residual
var shadowResidual = hillshadeNorm.subtract(nirNorm).rename("shadow_residual");

// Threshold residual
var thresh50 = shadowResidual.gt(0.5).selfMask();

// Export residual anomalies
Export.image.toDrive({
  image: thresh50,
  description: 'SunAligned_Terraces_P81',
  folder: 'GEE_Exports',
  fileNamePrefix: 'sun_aligned_terraces_p81',
  region: aoi,
  scale: 10,
  crs: 'EPSG:32644',
  maxPixels: 1e13
});
