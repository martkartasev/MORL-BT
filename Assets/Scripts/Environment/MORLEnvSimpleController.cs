using UnityEngine;

namespace Env5
{
    public class MORLEnvSimpleController : MonoBehaviour, EnvController
    {
        public Transform agent;
        public Transform goal;
        public Transform button;
        public Transform trigger;

        void Awake()
        {
            Initialize();
        }

        public bool AtGoal()
        {
            return agent.gameObject.GetComponent<CollisionDetector>().Touching(goal.gameObject);
        }

        public void Initialize()
        {
            agent.gameObject.GetComponent<Rigidbody>().velocity = Vector3.zero;
            goal.localPosition = new Vector3(Random.Range(-18, 18), 0.001f, Random.Range(-18, 18));
            agent.localPosition = new Vector3(Random.Range(-18, 18), 0.51f, Random.Range(-18, 18));
            if (button != null)
            {
                button.localPosition = new Vector3(Random.Range(-18, 18), 0.01f, Random.Range(-18, 18));
                while ((button.localPosition - goal.localPosition).magnitude < 4)
                {
                    button.localPosition = new Vector3(Random.Range(-18, 18), 0.01f, Random.Range(-18, 18));
                }
            }

            if (trigger != null)
            {
                agent.gameObject.GetComponent<ControlOther>().other = null;
                trigger.localPosition = new Vector3(Random.Range(-18, 18), 0.51f, Random.Range(-18, 18));
                while ((trigger.localPosition - button.localPosition).magnitude < 4 || (trigger.localPosition - agent.localPosition).magnitude < 4)
                {
                    trigger.localPosition = new Vector3(Random.Range(-18, 18), 0.51f, Random.Range(-18, 18));
                }
                trigger.GetComponent<Rigidbody>().velocity = Vector3.zero;
            }
        }

        public void Reset()
        {
            Initialize();
        }
    }
}