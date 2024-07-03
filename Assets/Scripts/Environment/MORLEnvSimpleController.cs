using UnityEngine;

namespace Env5
{
    public class MORLEnvSimpleController : MonoBehaviour, EnvController
    {
        public Transform agent;
        public Transform goal;
        
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
            goal.localPosition = new Vector3(Random.Range(-18,18), 0.001f, Random.Range(-18,18));
            agent.localPosition = new Vector3(Random.Range(-18,18), 0.06f, Random.Range(-18,18));
        }

        public void Reset()
        {
            Initialize();
        }
    }
}