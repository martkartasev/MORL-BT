using System.IO;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;
using UnityEngine;

namespace Env5
{
    public class MORLAgent : Agent
    {
        public MORLPlayerController playerController;
        public MORLEnvController envController;
        public IEnvActuator actuator = new EnvActuatorGrid5x5();
        public Camera envCamera;

        private void Start()
        {
            if (!Directory.Exists(Application.dataPath + "/Screenshots"))
            {
                Directory.CreateDirectory(Application.dataPath + "/Screenshots");
            }
        }

        public override void CollectObservations(VectorSensor sensor)
        {
            Vector3 playerPos = playerController.player.localPosition;
            Vector3 playerPosObs = playerPos / playerController.env.Width * 2f;
            sensor.AddObservation(playerPosObs);

            sensor.AddObservation(playerController.rb.velocity / playerController.maxSpeed);

            Vector3 triggerPos = playerController.env.trigger.localPosition;
            Vector3 distanceToTriggerObs = (triggerPos - playerPos) / playerController.env.Width;
            sensor.AddObservation(distanceToTriggerObs);

            Vector3 buttonPos = playerController.env.button.localPosition;
            Vector3 distanceToButtonObs = (buttonPos - playerPos) / playerController.env.Width;
            sensor.AddObservation(distanceToButtonObs);

            Vector3 button2Pos = playerController.env.goal.localPosition;
            Vector3 distanceTobutton2Obs = (button2Pos - playerPos) / playerController.env.Width;
            sensor.AddObservation(distanceTobutton2Obs);

            Vector3 distanceToBridgeObs = (playerController.env.BridgeEntranceLeft - playerPos) / playerController.env.Width;
            sensor.AddObservation(distanceToBridgeObs);
            Vector3 distanceToBridgeObs2 = (playerController.env.BridgeEntranceLeft2 - playerPos) / playerController.env.Width;
            sensor.AddObservation(distanceToBridgeObs2);

            int bridgeIsDown;
            bridgeIsDown = playerController.env.Button1Pressed() ? 1 : 0;
            sensor.AddObservation(bridgeIsDown);

            int triggerIsCarried;
            triggerIsCarried = playerController.IsControllingT1() ? 1 : 0;
            sensor.AddObservation(triggerIsCarried);
        }

        public override void Heuristic(in ActionBuffers actionsOut)
        {
            actuator.Heuristic(actionsOut);
        }

        public override void OnActionReceived(ActionBuffers actions)
        {
            playerController.ApplyAcceleration(actuator.GetAcceleration(actions) * 5);


            bool reset = actions.DiscreteActions[1] == 1;
            bool screenshot = actions.DiscreteActions[2] == 1;

            // if (reset || envController.AtGoal()) EndEpisode();
            if (reset) EndEpisode();
            if (screenshot) TakeScreenShot();
        }

        public override void OnEpisodeBegin()
        {
            envController.Reset();
        }

        public void TakeScreenShot()
        {
            if (envCamera == null) return;
            RenderTexture.active = envCamera.targetTexture;
            envCamera.Render();

            Texture2D imageOverview = new Texture2D(envCamera.targetTexture.width, envCamera.targetTexture.height, TextureFormat.RGB24, false);
            imageOverview.ReadPixels(new Rect(0, 0, envCamera.targetTexture.width, envCamera.targetTexture.height), 0, 0);
            imageOverview.Apply();

            byte[] bytes = imageOverview.EncodeToPNG();

            string filename = transform.parent.name + "_" + Time.fixedTime + ".png";

            var path = Application.dataPath + "/Screenshots/" + filename;
            File.WriteAllBytes(path, bytes);
        }
    }
}