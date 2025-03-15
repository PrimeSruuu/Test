// Set your Cesium Ion access token
Cesium.Ion.defaultAccessToken = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJqdGkiOiIyZWUyZmNiMS05ZDc4LTRmNjctYTJiNi01Y2M2MzhhNTdjMzAiLCJpZCI6Mjc1ODI3LCJpYXQiOjE3Mzk0NzExNjZ9.Qn9b6Vw2pHbDTdKLPmJoSLcG9W1QGocgKO7KGMfRwa8';

// Initialize Cesium Viewer with minimal UI
const viewer = new Cesium.Viewer('cesiumContainer', {
  animation: false,
  timeline: false,
  fullscreenButton: false,
  sceneModePicker: false,
  baseLayerPicker: false,
  homeButton: false,
  geocoder: false,
  navigationHelpButton: false,
  infoBox: false,
  selectionIndicator: false
});
viewer.cesiumWidget.creditContainer.style.display = "none";

// Set camera view (zoom out to view Earth)
viewer.camera.setView({
  destination: Cesium.Cartesian3.fromDegrees(0, 0, 20000000)
});

// Rotate Earth for a dynamic feel (adjust angular velocity as needed)
const angularVelocity = Cesium.Math.toRadians(0.02);
viewer.scene.postRender.addEventListener(() => {
  viewer.camera.rotate(Cesium.Cartesian3.UNIT_Z, angularVelocity);
});

// Endpoints for TLE data
const satelliteUrl = 'https://www.celestrak.org/NORAD/elements/gp.php?GROUP=active&FORMAT=tle';
const debrisUrl    = 'https://www.celestrak.org/NORAD/elements/gp.php?GROUP=cosmos-2251-debris&FORMAT=tle';

// Limits (you can change these values later)
const satelliteLimit = 200;
const debrisLimit = 200;

// ---------------------
// Fetch and Parse TLE Data
// ---------------------
async function fetchTLE(url) {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Error fetching data from ${url}: ${response.status}`);
  }
  return response.text();
}

function parseTLE(tleText) {
  // Split text into non-empty lines
  const lines = tleText.split('\n').filter(line => line.trim().length > 0);
  const tleSets = [];
  for (let i = 0; i < lines.length; i += 3) {
    if (i + 2 < lines.length) {
      tleSets.push({
        name: lines[i].trim(),
        line1: lines[i + 1].trim(),
        line2: lines[i + 2].trim()
      });
    }
  }
  return tleSets;
}

// ---------------------
// Compute Dynamic Position using satellite.js
// ---------------------
function computePosition(tle, date) {
  const satrec = satellite.twoline2satrec(tle.line1, tle.line2);
  const posVel = satellite.propagate(satrec, date);
  if (!posVel.position) return null;
  const gmst = satellite.gstime(date);
  const posGd = satellite.eciToGeodetic(posVel.position, gmst);
  return {
    longitude: satellite.degreesLong(posGd.longitude),
    latitude: satellite.degreesLat(posGd.latitude),
    height: posGd.height * 1000 // convert km to meters
  };
}

// ---------------------
// Compute Orbit Trajectory Points (one full orbit)
// ---------------------
function computeTrajectoryPoints(tle, stepSeconds = 60) {
  const satrec = satellite.twoline2satrec(tle.line1, tle.line2);
  // Mean motion (satrec.no) is in radians per minute.
  const periodSeconds = (2 * Math.PI / satrec.no) * 60;
  const positions = [];
  const startTime = new Date();
  // Sample positions along one full orbit
  for (let t = 0; t <= periodSeconds; t += stepSeconds) {
    const date = new Date(startTime.getTime() + t * 1000);
    const posVel = satellite.propagate(satrec, date);
    if (!posVel.position) continue;
    const gmst = satellite.gstime(date);
    const posGd = satellite.eciToGeodetic(posVel.position, gmst);
    const longitude = satellite.degreesLong(posGd.longitude);
    const latitude = satellite.degreesLat(posGd.latitude);
    const height = posGd.height * 1000; // km to meters
    positions.push(Cesium.Cartesian3.fromDegrees(longitude, latitude, height));
  }
  return positions;
}

// ---------------------
// Add Dynamic Entity (current position) & Orbit Path to Cesium
// ---------------------
function addEntity(tle, color) {
  // Dynamic moving point
  viewer.entities.add({
    name: tle.name,
    position: new Cesium.CallbackProperty((time, result) => {
      const pos = computePosition(tle, Cesium.JulianDate.toDate(time));
      return pos ? Cesium.Cartesian3.fromDegrees(pos.longitude, pos.latitude, pos.height) : result;
    }, false),
    point: {
      pixelSize: 10,
      color: color
    }
  });
}

function addOrbitPath(tle, color) {
  const positions = computeTrajectoryPoints(tle, 60); // sample every 60 seconds
  viewer.entities.add({
    name: tle.name + " Orbit Path",
    polyline: {
      positions: positions,
      width: 2,
      material: new Cesium.ColorMaterialProperty(color)
    }
  });
}

// ---------------------
// Load TLE Data and Render Entities & Orbits
// ---------------------
(async function loadAndRenderTLEs() {
  try {
    // Fetch TLE data concurrently for satellites and debris
    const [satText, debrisText] = await Promise.all([
      fetchTLE(satelliteUrl),
      fetchTLE(debrisUrl)
    ]);
    
    const satellites = parseTLE(satText).slice(0, satelliteLimit);
    const debris = parseTLE(debrisText).slice(0, debrisLimit);
    
    console.log("Loaded Satellites:", satellites);
    console.log("Loaded Debris:", debris);
    
    // For each satellite, add a dynamic entity and its orbital trajectory (yellow)
    satellites.forEach(tle => {
      addEntity(tle, Cesium.Color.YELLOW);
      addOrbitPath(tle, Cesium.Color.YELLOW);
    });
    
    // For each debris object, add a dynamic entity and its orbital trajectory (red)
    debris.forEach(tle => {
      addEntity(tle, Cesium.Color.RED);
      addOrbitPath(tle, Cesium.Color.RED);
    });
    
  } catch (error) {
    console.error("Error loading and rendering TLE data:", error);
  }
})();

// ==========================
// Interactive Object Selection and Collision Risk API Integration
// ==========================

// Fetch collision risk from your Flask API
async function fetchCollisionRisk(features) {
    const apiUrl = 'http://127.0.0.1:5000/predict'; // Updated API endpoint
    try {
      const response = await fetch(apiUrl, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(features)
      });
      if (!response.ok) {
        throw new Error(`API error: ${response.statusText}`);
      }
      const data = await response.json();
      return data.collision_risk;
    } catch (error) {
      console.error("Error fetching collision risk:", error);
      return null;
    }
  }
  
  // Show the info panel and populate it with basic details
  function showInfoPanel(entity) {
    const infoPanel = document.getElementById('infoPanel');
    document.getElementById('satName').textContent = entity.name || "Unknown Object";
    document.getElementById('satDetails').textContent = ""; // Clear or add additional details here
    updateInfoPanel("Loading collision risk...");
    infoPanel.classList.add('show');
  }
  
  // Update collision risk message in the info panel
  function updateInfoPanel(message) {
    document.getElementById('collisionRisk').textContent = message;
  }
  
  // Close button event for info panel
  document.getElementById('closeBtn').addEventListener('click', () => {
    document.getElementById('infoPanel').classList.remove('show');
  });
  
  // Set up object picking using Cesium's ScreenSpaceEventHandler
  const handler = new Cesium.ScreenSpaceEventHandler(viewer.scene.canvas);
  handler.setInputAction(async function(click) {
    const pickedObject = viewer.scene.pick(click.position);
    if (Cesium.defined(pickedObject) && pickedObject.id) {
      const entity = pickedObject.id;
      showInfoPanel(entity);
      
      // Safely access properties; use an empty object if undefined.
      const props = entity.properties || {};
      
      // Extract required features from the entity.
      const features = {
        sat1: props.sat1 ? props.sat1.getValue() : 0,
        sat2: props.sat2 ? props.sat2.getValue() : 0,
        altitude: props.altitude ? props.altitude.getValue() : 0,
        velocity: props.velocity ? props.velocity.getValue() : 0,
        inclination: props.inclination ? props.inclination.getValue() : 0,
        eccentricity: props.eccentricity ? props.eccentricity.getValue() : 0,
        raan: props.raan ? props.raan.getValue() : 0,
        perigee: props.perigee ? props.perigee.getValue() : 0,
        anomaly: props.anomaly ? props.anomaly.getValue() : 0
      };
      
      // Fetch collision risk from Flask API
      const risk = await fetchCollisionRisk(features);
      if (risk !== null) {
        updateInfoPanel(`Collision Risk: ${risk}`);
      } else {
        updateInfoPanel("Collision Risk: Unavailable");
      }
    }
  }, Cesium.ScreenSpaceEventType.LEFT_CLICK);
  