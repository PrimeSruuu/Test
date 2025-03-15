// Set your Cesium Ion access token
Cesium.Ion.defaultAccessToken = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJqdGkiOiIyZWUyZmNiMS05ZDc4LTRmNjctYTJiNi01Y2M2MzhhNTdjMzAiLCJpZCI6Mjc1ODI3LCJpYXQiOjE3Mzk0NzExNjZ9.Qn9b6Vw2pHbDTdKLPmJoSLcG9W1QGocgKO7KGMfRwa8';

// Initialize Cesium Viewer
const viewer = new Cesium.Viewer('cesiumContainer', {
    animation: false,  // Remove timeline
    timeline: false,   // Remove Cesium's default timeline
    fullscreenButton: false, 
    sceneModePicker: false, 
    baseLayerPicker: false,
    homeButton: false, 
    geocoder: false, 
    navigationHelpButton: false,
    infoBox: false, 
    selectionIndicator: false
});

viewer.cesiumWidget.creditContainer.style.display = "none"; // Hide Cesium's default credits

// Zoom to Earth
viewer.camera.setView({
    destination: Cesium.Cartesian3.fromDegrees(0, 0, 20000000) // Adjust zoom level
});

// Set up Earth's rotation speed (radians per frame)
const angularVelocity = Cesium.Math.toRadians(0.02);
function rotateEarth() {
  viewer.scene.postRender.addEventListener(() => {
    viewer.camera.rotate(Cesium.Cartesian3.UNIT_Z, angularVelocity);
  });
}
rotateEarth();