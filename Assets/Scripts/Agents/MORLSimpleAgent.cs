using System;
using Unity.MLAgents.Sensors;
using Unity.VisualScripting;
using UnityEngine;

namespace Env5
{
    public class MORLSimpleAgent : MORLAgent
    {
        public Transform player;
        public Transform goal;

        public Rigidbody trigger;
        public Transform button;
        private CollisionDetector collisionDetector;
        private ControlOther controlTrigger;

        private void Start()
        {
            base.Start();
            collisionDetector = GetComponent<CollisionDetector>();
            controlTrigger = GetComponent<ControlOther>();
        }

        public override void CollectObservations(VectorSensor sensor)
        {
            Vector3 playerPos = player.localPosition;

            sensor.AddObservation(playerPos);
            sensor.AddObservation(goal.localPosition - playerPos);

            if (!velocityBased) sensor.AddObservation(rb.velocity);

            if (trigger != null)
            {
                sensor.AddObservation(trigger.transform.localPosition - playerPos);
                sensor.AddObservation(button.transform.localPosition - playerPos);
                sensor.AddObservation(controlTrigger.other != null ? 1 : 0);
                sensor.AddObservation(button.gameObject.GetComponent<CollisionDetector>().Touching("Target") ? 1 : 0);
            }
        }

        public void FixedUpdate()
        {
            if (trigger != null && collisionDetector.Touching("Target") && !button.gameObject.GetComponent<CollisionDetector>().Touching("Target") && !button.gameObject.GetComponent<CollisionDetector>().Touching("Player"))
            {
                controlTrigger.other = trigger;
            }

            if (trigger != null && controlTrigger.other == trigger && collisionDetector.Touching("Button"))
            {
                controlTrigger.other = null;
                trigger.position = new Vector3(button.position.x, button.position.y + 0.5f, button.position.z);
                trigger.velocity = Vector3.zero;
            }
        }
    }
}