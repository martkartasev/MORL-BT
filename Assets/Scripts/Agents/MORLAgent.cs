using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;
using UnityEngine;
using UnityEngine.Serialization;

namespace Env5
{
    public class MORLAgent : Agent
    {
        public MORLPlayerController playerController;
        public MORLEnvController envController;
        public IEnvActuator actuator = new EnvActuatorGrid5x5();

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
        }

        public override void Heuristic(in ActionBuffers actionsOut)
        {
            actuator.Heuristic(actionsOut);
        }

        public override void OnActionReceived(ActionBuffers actions)
        {
            playerController.ApplyAcceleration(actuator.GetAcceleration(actions) * 5);

            bool reset = actions.DiscreteActions[1] == 1;
            if (reset || envController.AtGoal()) EndEpisode();
        }

        public override void OnEpisodeBegin()
        {
            envController.Reset();
        }
    }
}