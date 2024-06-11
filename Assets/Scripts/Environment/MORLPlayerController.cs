using UnityEngine;

namespace Env5
{
    public class MORLPlayerController : MonoBehaviour
    {
        public Transform player;
        public Rigidbody rb;
        public MORLEnvController env;
        private float closenessDistance = 3.0f;
        private float maxAcc = 10f;
        internal float maxSpeed = 10f;
        public bool safelyPlaceTrigger;

        public float MaxAcc { get => maxAcc; }

        public void ApplyAcceleration(Vector3 acceleration)
        {
            rb.AddForce(acceleration, ForceMode.Acceleration);

            if (rb.velocity.magnitude > maxSpeed)
            {
                rb.velocity = rb.velocity.normalized * maxSpeed;
            }
        }

        void OnCollisionStay(Collision collision)
        {
            if (collision.gameObject.tag == "Target")
            {
                if (!env.Button1Pressed() && !player.GetComponent<CollisionDetector>().Touching("Button"))
                {
                    StartControlTrigger();
                }
            }
            else if (collision.gameObject.tag == "Button")
            {
                StopControlTrigger1(true);
            }
        }

        public void StopControl()
        {
            StopControlTrigger1();
        }

        private void StopControlTrigger1(bool press = false)
        {
            if (IsControllingT1())
            {
                ControlOther controlOther = GetComponent<ControlOther>();
                controlOther.enabled = false;

                if (safelyPlaceTrigger)
                {
                    Vector2 safePoint = FindSafePoint(player.position, rb.velocity, env.button.position);
                    env.trigger.position = new Vector3(safePoint.x, env.button.position.y + 0.5f, safePoint.y);
                }
                else
                {
                    env.trigger.position = new Vector3(env.button.position.x, env.button.position.y + 0.5f, env.button.position.z);
                }
                env.trigger.GetComponentInParent<Rigidbody>().velocity = Vector3.zero;
            }
        }

        private static Vector2 FindSafePoint(Vector3 c1_3d, Vector3 r1_3d, Vector3 c2_3d)
        {
            var c1 = new Vector2(c1_3d.x, c1_3d.z);
            var r1 = new Vector2(r1_3d.x, r1_3d.z).normalized;
            var c2 = new Vector2(c2_3d.x, c2_3d.z);

            var t1 = Vector2.Dot(c2 - c1, r1);
            var p = c1 + t1 * r1;
            var d2 = p - c2;
            var r2 = d2.normalized;
            var x = p - r2 * 1.5f;
            return x;
        }

        public bool IsControllingT1()
        {
            ControlOther controlOther = GetComponent<ControlOther>();
            return controlOther.enabled && controlOther.other == env.trigger.GetComponent<Rigidbody>();
        }

        
        private void StartControlTrigger()
        {
            ControlOther controlOther = GetComponent<ControlOther>();
            controlOther.enabled = true;
            controlOther.other = env.trigger.GetComponent<Rigidbody>();
        }
    
    }
}
