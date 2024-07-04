using Unity.MLAgents.Sensors;
using UnityEngine;

namespace Env5
{
    public class MORLSimpleAgent : MORLAgent
    {
        public Transform player;
        public Transform goal;

        public override void CollectObservations(VectorSensor sensor)
        {
            Vector3 playerPos = player.localPosition;
            Vector3 playerPosObs = playerPos;
            sensor.AddObservation(playerPosObs);

            Vector3 button2Pos = goal.localPosition;
            Vector3 distanceTobutton2Obs = (button2Pos - playerPos);
            sensor.AddObservation(distanceTobutton2Obs);
            
            if(!velocityBased) sensor.AddObservation(rb.velocity);
        }
    }
}